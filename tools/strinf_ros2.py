import _init_path
import argparse
import datetime
import os
import time
import json
from pathlib import Path

import torch
import gc
import sys
import pickle

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, MultiArrayDimension
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.time import Time
from rclpy.clock import ClockType, Clock
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from autoware_perception_msgs.msg import TrackedObjects, DetectedObjects, DetectedObject, ObjectClassification
from valo_msgs.msg import Float32MultiArrayStamped
from callback_profile.msg import CallbackProfile
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from multiprocessing import Process, Barrier, Pool, Value
import concurrent
from torch.profiler import profile, record_function, ProfilerActivity
from pcdet.models.model_utils.tensorrt_utils.trtwrapper import TRTWrapper

from pcdet.utils.ros2_utils import *

# export FINE_GRAINED_EVAL=1 for fine grained evaluation results saved to json

VALO_DEBUG = False
DO_DYN_SCHED = True
ALWAYS_BLOCK_SCHED = False
EVAL_TRACKER = False
ANYTIME_CAPABLE = True
ENABLE_TILE_DROP = False
VISUALIZE = False
PROFILE = False
DO_EVAL = False
USE_EGOVEL_FOR_DL_PRED = False

assert (DO_DYN_SCHED != ALWAYS_BLOCK_SCHED)

class StreamingEvaluator(Node):
    def __init__(self, args, period_sec, shr_tracker_time_sec):
        super().__init__('streaming_evaluator')

        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.period_sec = period_sec
        self.shr_tracker_time_sec = shr_tracker_time_sec

        # receive objects and update the buffer
        self.all_tracked_objects = [] #TrackedObjects()]
        self.all_detected_objects = []
        self.sampled_indices = []

        self.cb_group = rclpy.callback_groups.ReentrantCallbackGroup()
        #self.cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.cmd_sub = self.create_subscription(Header, 'evaluator_cmd',
                self.cmd_callback, 10) #, callback_group=self.cb_group)

        if EVAL_TRACKER:
            self.tracker_sub = self.create_subscription(TrackedObjects, 'tracked_objects',
                    self.tracker_callback, 10, callback_group=self.cb_group)
            self.tracker_exec_time_sub = self.create_subscription(CallbackProfile,
                    '/multi_object_tracker/assoc_cb_profile', self.tracker_exectime_callback,
                    1, callback_group=self.cb_group)
        else:
            self.detector_sub = self.create_subscription(Float32MultiArrayStamped, 'detected_objects_raw',
                    self.detector_callback, 10, callback_group=self.cb_group)

        if VISUALIZE:
            self.det_debug_publisher = self.create_publisher(DetectedObjects, 'detected_objects_debug', 1)
            self.ground_truth_publisher = self.create_publisher(DetectedObjects, 'ground_truth_objects', 1)

            # For debug
            qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10)
            self.pc_publisher = self.create_publisher(PointCloud2, 'point_cloud', qos_profile)

            self.pc_sub = self.create_subscription(Header, 'point_cloud_idx',
                    self.publish_pc, 10, callback_group=self.cb_group)

        cfg_from_yaml_file(args.cfg_file, cfg)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)
        self.cfg = cfg

        logger, test_set = get_dataset(cfg)
        oc = ObjectClassification()
        cls_names = cfg.CLASS_NAMES
        self.cls_mapping = { cls_names.index(name)+1: oc.__getattribute__(name.upper()) \
                for name in cls_names }
        self.inv_cls_mapping = {v:k for k,v in self.cls_mapping.items()}

        self.dataset = test_set
        num_samples = len(self.dataset)
        print('[SE] Num dataset elems:', num_samples)

        if VISUALIZE:
            numproc=8
            self.debug_points = [None] * num_samples
            self.gt_objects= [None] * num_samples
            splits = np.array_split(np.arange(num_samples), numproc)

            print('[SE] Loading debug points and ground truth')
            load_start_time = time.time()
            with concurrent.futures.ProcessPoolExecutor(max_workers=numproc) as executor:
                futures = [executor.submit(get_debug_pts_and_gt_objects, split.tolist(), \
                        self.dataset, self.cls_mapping) for split in splits]
                for fut, split in zip(futures, splits):
                    debug_pts, gt_objects = fut.result()
                    for c_i, s_i in enumerate(split):
                        self.debug_points[s_i] = debug_pts[c_i]
                        self.gt_objects[s_i] = gt_objects[c_i]
            load_end_time = time.time()
            print('[SE] Loaded debug points and ground truth, it took', \
                    load_end_time-load_start_time, 'seconds')

        with open('token_to_pos.json', 'r') as file:
            self.token_to_pos = json.load(file)

        self.br = tf2_ros.TransformBroadcaster(self)
        self.time_to_sub = Time(seconds=0, nanoseconds=0)
        self.reset = False

    def pose_pub_timer_callback(self):
        sec, nanosec = self.time_to_sub.seconds_nanoseconds()
        init_time_shifted = self.init_time + sec + (nanosec / 1e9)
        i = int((time.time() - init_time_shifted) / self.period_sec)
        if i < len(self.dataset):
            stamp_time = init_time_shifted + self.period_sec * i
            cur_stamp = seconds_to_TimeMsg(stamp_time)
            pose = self.token_to_pos[self.dataset.infos[i]['token']]
            tf_ep = pose_to_tf(pose['ep_translation'], pose['ep_rotation'], cur_stamp, 'world', 'body')
            tf_cs = pose_to_tf(pose['cs_translation'], pose['cs_rotation'], cur_stamp, 'body', 'base_link')
            self.br.sendTransform(tf_ep)
            self.br.sendTransform(tf_cs)

    def publish_pc(self, msg):
        # Following takes 5 ms
        idx = int(msg.frame_id)
        if idx < len(self.dataset) and self.debug_points[idx] is not None:
            pc2_msg = self.debug_points[idx]
            pc2_msg.header.stamp = msg.stamp
            pc2_msg.header.frame_id = 'base_link'
            self.pc_publisher.publish(pc2_msg)
            self.debug_points[idx] = None # clear mem
            self.pc_pub_idx = idx + 1

            #publish ground truth as well
            gt_objs = self.gt_objects[idx]
            if gt_objs is not None:
                gt_objs.header.stamp = msg.stamp
                self.ground_truth_publisher.publish(gt_objs)

    def tracker_callback(self, msg):
         # save the time of msg arrival for streaming eval
        duration = (self.system_clock.now() - self.time_to_sub).to_msg()
        msg.header.stamp = TimeMsg(sec=duration.sec, nanosec=duration.nanosec)
        self.all_tracked_objects.append(msg)

    def tracker_exectime_callback(self, msg):
        start, end = Time.from_msg(msg.start_stamp), Time.from_msg(msg.end_stamp)
        dur = (end-start).to_msg()
        self.shr_tracker_time_sec.value = dur.sec + (dur.nanosec * 1e-9)

    def detector_callback(self, msg):
        duration = (self.system_clock.now() - self.time_to_sub).to_msg()
        msg.header.stamp = TimeMsg(sec=duration.sec, nanosec=duration.nanosec)

        num_obj = msg.array.layout.dim[0].size
        if num_obj > 0:
            self.all_detected_objects.append(msg)
            if VISUALIZE:
                detected_objs = f32_multi_arr_to_detected_objs(msg, self.cls_mapping)
                self.det_debug_publisher.publish(detected_objs)

    def start_sampling(self, bar):
        self.bar = bar
        self.bar.wait()
        self.init_time = time.time()
        print('[SE] Started sampling at', self.init_time)

        self.pose_timer = self.create_timer(self.period_sec/2, self.pose_pub_timer_callback,
                callback_group=self.cb_group, clock=self.system_clock)

    # Receive buffer reset and start timer commands
    def cmd_callback(self, msg):
        cmd = msg.frame_id
        if cmd == 'stop_sampling': # stop timer and evaluate
            if PROFILE:
                self.bar.wait()
                return

            if EVAL_TRACKER:
                sampled_objects, frame_id = self.get_streaming_eval_samples(self.all_tracked_objects)
                print(f'[SE] Sampled {len(self.all_tracked_objects)} tracked objects')
            else:
                sampled_objects, frame_id = self.get_streaming_eval_samples(self.all_detected_objects)
                print(f'[SE] Sampled {len(self.all_detected_objects)} detected objects')

            self.do_eval(sampled_objects, frame_id)
            self.bar.wait()
        elif cmd == 'time_fix':
            sec, nanosec = self.time_to_sub.seconds_nanoseconds()
            sec += msg.stamp.sec
            nanosec += msg.stamp.nanosec
            if nanosec >= int(1e9):
                nanosec -= int(1e9)
                sec += 1
            self.time_to_sub = Time(seconds=sec, nanoseconds=nanosec)
        elif cmd == 'reset':
            self.reset = True

    def get_streaming_eval_samples(self, all_objects):
        # Streaming eval
        #Now do manual sampling base on time
        init_sec = int(math.floor(self.init_time))
        init_ns = int((self.init_time - init_sec) * 1e9)
        times_ns = []
        for tobjs in all_objects:
            sec = tobjs.header.stamp.sec - init_sec
            assert sec >= 0
            times_ns.append(sec * int(1e9) + tobjs.header.stamp.nanosec)
        times_ns = np.array(times_ns)

        obj_cls = type(all_objects[0])

        sampled_objects = []
        num_ds_elems = len(self.dataset)
        period_ns = int(self.period_sec * 1e9)
        for i in range(num_ds_elems):
            sample_time_ns = init_ns + i*period_ns
            aft = times_ns > sample_time_ns
            if aft[0] == True:
                sampled_objects.append(obj_cls())
                #print('0', end=' ')
            elif aft[-1] == False:
                sampled_objects.append(all_objects[-1])
                #print('-1', end=' ')
            else:
                sample_idx = np.argmax(aft) - 1
                sampled_objects.append(all_objects[sample_idx])
                #print(sample_idx, end=' ')
            #if i % 40 == 0:
                #print()
        print()

        num_sampled = len(sampled_objects)
        return sampled_objects, all_objects[-1].header.frame_id

    def get_eval_samples(self,  all_objects):
        num_ds_elems = len(self.dataset)
        while len(all_objects) < num_ds_elems:
            all_objects.append(all_objects[-1])
        return all_objects[:num_ds_elems]

    def do_eval(self, sampled_objects, frame_id):
        #Convert them to openpcdet format

        det_annos = []
        num_ds_elems = len(self.dataset)
        for i in range(num_ds_elems):
            data_dict = self.dataset.get_metadata_dict(i)
            for k, v in data_dict.items():
                data_dict[k] = [v] # make it a batch dict
            objs = sampled_objects[i]

            if isinstance(objs, DetectedObjects) or isinstance(objs, TrackedObjects):
                num_objs = len(objs.objects)
                boxes, scores, labels = torch.empty((num_objs, 9)), torch.empty(num_objs), torch.empty(num_objs, dtype=torch.long)

                mask = torch.ones(num_objs, dtype=torch.bool)
                for j, obj in enumerate(objs.objects):
                    scores[j] = obj.existence_probability
                    labels[j] = self.inv_cls_mapping[obj.classification[0].label]
                    try:
                        boxes[j,0] = obj.kinematics.pose_with_covariance.pose.position.x
                        boxes[j,1] = obj.kinematics.pose_with_covariance.pose.position.y
                        boxes[j,2] = obj.kinematics.pose_with_covariance.pose.position.z
                        boxes[j,3] = obj.shape.dimensions.x
                        boxes[j,4] = obj.shape.dimensions.y
                        boxes[j,5] = obj.shape.dimensions.z
                        quat = obj.kinematics.pose_with_covariance.pose.orientation
                        boxes[j,6] = euler_from_quaternion((quat.x, quat.y, quat.z, quat.w))[2] # yaw
                        linear_x = obj.kinematics.twist_with_covariance.twist.linear.x
                        boxes[j,7] = linear_x * math.cos(boxes[j,6])
                        boxes[j,8] = linear_x * math.sin(boxes[j,6])

                        if torch.any(torch.isnan(boxes[j, :])):
                            mask[j] = False
                            print(f'[SE] Warning, object [{i},{j}] has nan in it, ignoring')
                    except RuntimeError:
                        # probably floating point conversion error
                        print(f'[SE] Warning, object [{i},{j}] has some problem, ignoring')
                        mask[j] = False
                data_dict['final_box_dicts'] = [{
                            'pred_boxes': boxes[mask],
                            'pred_scores': scores[mask],
                            'pred_labels': labels[mask]
                }]
            else: # Float32 arr
                data_dict['final_box_dicts'] =  [f32_multi_arr_to_pred_dict(objs)]

            det_annos += self.dataset.generate_prediction_dicts(
                data_dict, data_dict['final_box_dicts'], self.dataset.class_names, output_path=None
            )

        calib_dl = float(os.getenv('CALIB_DEADLINE_MILLISEC', 0.))
        eval_d = {
            'cfg': self.cfg,
            'det_annos': det_annos,
            'annos_in_glob_coords':(frame_id != 'base_link'),
            'calib_deadline_ms': calib_dl}

        if DO_EVAL:
            #nusc_annos = {} # not needed but keep it anyway
            result_str, result_dict = self.dataset.evaluation(
                det_annos, self.dataset.class_names,
                eval_metric=self.cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path='./tmp_results',
                #nusc_annos_outp=nusc_annos,
                boxes_in_global_coords=(frame_id != 'base_link'),
                #det_elapsed_musec=det_elapsed_musec,
            )

            print(result_str)
            eval_d['result_str'] = result_str

        with open(f'eval_data_{calib_dl}ms.pkl', 'wb') as f:
            pickle.dump(eval_d, f)

class InferenceServer(Node):
    def __init__(self, args, period_sec):
        super().__init__('inference_server')
        #self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)])
        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        self.init_model(args)

        self.period_sec = period_sec

        self.det_publisher = self.create_publisher(Float32MultiArrayStamped, 'detected_objects_raw', 1)

        self.cmd_publisher = self.create_publisher(Header, 'evaluator_cmd', 10)

        self.tcount = self.model_cfg.MODEL.get('TILE_COUNT', 0)
        print('[IS] TILE COUNT:', self.tcount)

        #if VISUALIZE:
        self.pc_idx_publisher = self.create_publisher(Header, 'point_cloud_idx', 10)
        self.rect_array_pub = self.create_publisher(MarkerArray, 'chosen_tiles', 10)

        pc_range =  self.dataset.point_cloud_range
        if self.tcount == 0:
            vertices_xyz = np.array((
                (pc_range[0], pc_range[1], 0.), #-x -y
                (pc_range[3], pc_range[1], 0.), #+x -y
                (pc_range[3], pc_range[4], 0.), #+x +y
                (pc_range[0], pc_range[4], 0.)  #-x +y
            ))
            whole_area_rect = gen_viz_filled_rectangle(vertices_xyz, 0)
            whole_area_rect.header.stamp = self.system_clock.now().to_msg()

            ma = MarkerArray()
            ma.markers.append(whole_area_rect)
            self.rect_array_pub.publish(ma)

        if VISUALIZE and self.tcount > 0:
            # To publish chosen tiles
            tile_w = (pc_range[3] - pc_range[0]) / self.tcount
            #tile_h = (pc_range[4] - pc_range[1])
            v_top = np.array((pc_range[0], pc_range[1], 0.))
            v_bot = np.array((pc_range[0], pc_range[4], 0.))
            
            self.rect_arr = MarkerArray()
            shift = np.array((tile_w, 0., 0.))
            stamp = self.system_clock.now().to_msg()
            for i in range(self.tcount):
                vertices = (v_top, v_top + shift, v_bot + shift, v_bot)
                rect = gen_viz_filled_rectangle(vertices, i)
                rect.header.stamp = stamp
                self.rect_arr.markers.append(rect)
                v_top += shift
                v_bot += shift

            self.rect_array_pub.publish(self.rect_arr)

    def init_model(self, args):
        cfg_from_yaml_file(args.cfg_file, cfg)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)

        logger, test_set = get_dataset(cfg)

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.eval()
        model.cuda()
        model.enable_tile_drop = ENABLE_TILE_DROP
        self.model_cfg = cfg

        self.dataset = model.dataset
        self.num_samples = len(self.dataset)
        self.batch_dicts_arr = [None] * self.num_samples

        oc = ObjectClassification()
        cls_names = cfg.CLASS_NAMES
        self.cls_mapping = { cls_names.index(name)+1: oc.__getattribute__(name.upper()) \
                for name in cls_names }

        tkn = model.token_to_scene[self.dataset.infos[0]['token']]
        self.scene_start_indices = [(tkn, 0)]
        for i in range(1, self.num_samples):
            new_tkn = model.token_to_scene[self.dataset.infos[i]['token']]
            if new_tkn != tkn:
                self.scene_start_indices.append((new_tkn, i))
            tkn = new_tkn
        self.scene_start_indices.append(('', self.num_samples))
        print('[IS] Scene start indices:', self.scene_start_indices)

        self.calib_dl = float(os.getenv('CALIB_DEADLINE_MILLISEC', 0.))

        print('[IS] Calibrating...')
        torch.cuda.cudart().cudaProfilerStop()
        with torch.no_grad():
            model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

        # Remove the hooks now
        model.pre_hook_handle.remove()
        model.post_hook_handle.remove()

        self.model = model

        global ANYTIME_CAPABLE
        if ANYTIME_CAPABLE:
            # load trt model
            power_mode = os.getenv('PMODE', 'pmode_0000')
            trt_path = f"./deploy_files/trt_engines/{power_mode}/deadline_pred_mdl.engine"
            print('Trying to load trt engine at', trt_path)
            try:
                self.dl_pred_trt = TRTWrapper(trt_path, ['bev_inp'], ['deadline'])
                self.dl_pred_out_buf = None
            except:
                print('TensorRT wrapper for deadline pred model throwed exception')
                ANYTIME_CAPABLE = False

    def load_scene(self, scene_token):
        for idx, (s_tkn, ind) in enumerate(self.scene_start_indices):
            if s_tkn == scene_token:
                ind_range = np.arange(ind, self.scene_start_indices[idx+1][1])
                print('Loading samples ', ind_range[0], '...', ind_range[-1])

                numproc=4
                splits = np.array_split(ind_range, numproc)
                with concurrent.futures.ThreadPoolExecutor(max_workers=numproc) as executor:
                    futures = [executor.submit(collate_dataset, split, self.dataset) \
                            for split in splits]
                    for fut, split in zip(futures, splits):
                        chunk = fut.result()
                        for c_i, s_i in enumerate(split):
                            self.batch_dicts_arr[s_i] = chunk[c_i]

                if idx > 0:
                    prev_ind = self.scene_start_indices[idx-1][1]
                    for d in range(prev_ind, ind):
                        self.batch_dicts_arr[d] = None
                print('Samples loaded')
                return

    def get_dyn_deadline_sec(self, last_sample_idx, sample_idx):
        if self.calib_dl != 0.:
            return self.calib_dl / 1000.

        if not ANYTIME_CAPABLE:
            return 10.0 # ignore deadline

#        if last_sample_idx != -1:
#            deadlines_ms = [95, 145, 195]
#            chosen_dl = self.model.calc_bbox_misalignment(last_sample_idx, sample_idx, deadlines_ms)
#            return chosen_dl / 1000.0 # make it sec 

        if USE_EGOVEL_FOR_DL_PRED:
            max_vel_n = 15 # meters per second
            max_deadline = 0.245
            min_deadline = 0.095
            if sample_idx > 0 and sample_idx < self.num_samples:
                vel = self.model.get_egovel(sample_idx)
                vel_n = np.linalg.norm(vel)
                return max_deadline - (vel_n/max_vel_n) * (max_deadline - min_deadline)
            else:
                return max_deadline
        else:
            return 10.0

    def infer_loop(self, bar, shr_tracker_time_sec):
        model = self.model
        tdiffs = [0. for i in range(len(self.dataset))]

        # Disable garbage collection to make things deterministic
        # pytorchs cuda allocator should cleanup its own memory seperately from gc
        # NOTE I hope dram won't overflow!
        gc.disable()
        dummy_tensor = torch.empty(1024*1024*1024, device='cuda')
        torch.cuda.synchronize()
        del dummy_tensor

        gc.collect()
        bar.wait() # sync with timer

        init_time = time.time()
        print('[IS] Starting inference at', init_time)
        i, last_scene_token = -1, ''
        #wakeup_time = init_time + self.period_sec
        self.model.last_elapsed_time_musec = 100000
        processed_inds = set()
        last_processed_ind = -1

        #if self.calib_dl > 0. and self.calib_dl < self.period_sec*1000:
        #    DO_DYN_SCHED = False
        #    ALWAYS_BLOCK_SCHED = True
        model.latest_batch_dict = None

        while rclpy.ok():
            if VALO_DEBUG:
                i += 1
            else:
                sched_time = time.time()
                diff = sched_time - init_time
                cur_tail = diff - math.floor(diff / self.period_sec) * self.period_sec

                if ALWAYS_BLOCK_SCHED and cur_tail > 0.001: # tolerate 1 ms error
                    time.sleep(self.period_sec - cur_tail + 0.001)
                elif DO_DYN_SCHED:
                    # Dynamic scheduling
                    # Calculate current and next tail
                    #if ANYTIME_CAPABLE:
                    #    exec_time_sec = dyn_deadline_sec
                    #else:
                    exec_time_sec = model.last_elapsed_time_musec * 1e-6

                    if EVAL_TRACKER:
                        exec_time_sec += shr_tracker_time_sec.value
                    pred_finish_time = sched_time + exec_time_sec
                    diff = pred_finish_time - init_time
                    next_tail = diff - math.floor(diff / self.period_sec) * self.period_sec
                    if next_tail < cur_tail:
                        # Extra 1 ms is added to make sure slept time is enough
                        time.sleep(self.period_sec - cur_tail + 0.001)
                cur_time = time.time()
                i = int((cur_time- init_time) / self.period_sec)

                if i in processed_inds:
                    i_ = i
                    while i in processed_inds:
                        i += 1
                    target_time = init_time + i * self.period_sec
                    sleep_time = target_time - cur_time + 0.001
                    print(f"[IS] Trying to process the sample {i_} again, skipping to"
                            f" {i} by sleeping {sleep_time} seconds.")
                    time.sleep(sleep_time)

            if PROFILE and i > 200 or i >= self.num_samples:
                break


            sample_token = self.dataset.infos[i]['token']
            scene_token = self.model.token_to_scene[sample_token]
            if scene_token != last_scene_token:
                # Load scene to memory
                scene_load_start_t = time.time()
                self.load_scene(scene_token)

                gc.collect()
                #print(torch.cuda.memory_summary())
                scene_load_time = time.time() - scene_load_start_t
                init_time+= scene_load_time # race condition, but its ok
                #wakeup_time += scene_load_time

                # Notify Evaluator
                eval_cmd = Header()
                eval_cmd.frame_id = 'time_fix'
                sec = int(math.floor(scene_load_time))
                eval_cmd.stamp = TimeMsg(sec=sec, nanosec=int((scene_load_time-sec)*1e9))
                self.cmd_publisher.publish(eval_cmd)

                last_processed_ind = -1
                model.latest_batch_dict = None

            last_scene_token = scene_token
            dyn_deadline_sec = self.get_dyn_deadline_sec(last_processed_ind, i)

            with record_function("inference"):
                batch_dict = self.batch_dicts_arr[i]
                self.batch_dicts_arr[i] = None
                model.measure_time_start('End-to-end')
                model.measure_time_start('PreProcess')
                start_time = time.time()

                # Tell Evaluator to publish point cloud for debug
                stamp_time = init_time + self.period_sec * i
                cur_stamp = seconds_to_TimeMsg(stamp_time)

                if VISUALIZE:
                    idx_msg = Header()
                    idx_msg.frame_id = str(i)
                    idx_msg.stamp = cur_stamp
                    self.pc_idx_publisher.publish(idx_msg)

                deadline_sec_override, reset = model.initialize(sample_token)
                #if reset:
                    #Clear buffers
                #    model.latest_batch_dict = None

                with torch.no_grad():
                    load_data_to_gpu(batch_dict)

#                    pts = batch_dict['points']
#                    print(torch.unique(pts[:, -2].long()))
#                    for i in range(6):
#                        print(pts[:, i].min(), pts[:, i].max())

                    batch_dict['scene_reset'] = reset
                    batch_dict['start_time_sec'] = start_time
                    batch_dict['deadline_sec'] = dyn_deadline_sec #float(self.model_cfg.MODEL.DEADLINE_SEC)
                    batch_dict['abs_deadline_sec'] = start_time + batch_dict['deadline_sec']
                    model.measure_time_end('PreProcess')
                    
#                    frame_id <class 'numpy.ndarray'> # can forward without it
#                    metadata <class 'numpy.ndarray'> # can forward without it
#                    points torch.Size([274237, 6])
#                    batch_size <class 'int'>
#                    scene_reset <class 'bool'> # set it false
#                    start_time_sec <class 'float'> #
#                    deadline_sec <class 'float'>
                    batch_dict = model.forward(batch_dict)

                    model.measure_time_start('PostProcess')
                    if 'final_box_dicts' in  batch_dict:
                        if 'pred_ious' in batch_dict['final_box_dicts'][0]:
                            del batch_dict['final_box_dicts'][0]['pred_ious']
                        for k,v in batch_dict['final_box_dicts'][0].items():
                            batch_dict['final_box_dicts'][0][k] = v.cpu()

                model.latest_batch_dict = {k: batch_dict[k] for k in \
                        ['final_box_dicts', 'metadata']}

                for k in ('chosen_tile_coords', 'nonempty_tile_coords'):
                    if k in batch_dict:
                        model.latest_batch_dict[k] = batch_dict[k]

                torch.cuda.synchronize()
                model.measure_time_end('PostProcess')
                model.measure_time_end('End-to-end')
                finish_time = time.time()

                model.calc_elapsed_times() 
                model.last_elapsed_time_musec = int(model._time_dict['End-to-end'][-1] * 1000)

                tdiffs[i] = round(finish_time - batch_dict['abs_deadline_sec'], 3)

                if VALO_DEBUG and 'forecasted_dets' in batch_dict:
                    pred_dict = batch_dict['forecasted_dets'][0]
                    pred_dict['pred_labels'] += 1
                    print('Publishing forecasted dets')
                else:
                    pred_dict = batch_dict['final_box_dicts'][0]

                if VISUALIZE and 'chosen_tile_coords' in batch_dict:
                    # Chosen tile coords is given, visualize the regions as rectangles
                    ctc = batch_dict['chosen_tile_coords']
                    for i in range(len(self.rect_arr.markers)):
                        self.rect_arr.markers[i].header.stamp = cur_stamp
                        self.rect_arr.markers[i].action = Marker.ADD if i in ctc else Marker.DELETE
                    self.rect_array_pub.publish(self.rect_arr)

                self.publish_dets(pred_dict, cur_stamp)

                finishovski_time = time.time()
                #Following prints 0.6 ms
                #print('Finishiovski took:', round((finishovski_time-finish_time)*1000, 2), 'ms')

                #sleep_time = wakeup_time - time.time()
                #if sleep_time > .0:
                #    time.sleep(sleep_time + 0.001)
                #else:
                #    print('[IS] Overrun', round(sleep_time * 1000, 2), 'ms')
                #wakeup_time += self.period_sec

                #if i % 10 == 0:
                #    gc.collect()
                #    print(torch.cuda.memory_summary())

            processed_inds.add(i)
            last_processed_ind = i

        eval_cmd = Header()
        eval_cmd.frame_id = 'stop_sampling'
        self.cmd_publisher.publish(eval_cmd)

        gc.enable()
        #for i, tdiff in enumerate(tdiffs):
        #    if tdiff > 0:
        #        print(f'Deadline {i} missed with {tdiff * 1000} ms')
        model.print_time_stats()


    # This func takes less than 1 ms, ~0.6 ms
    def publish_dets(self, pred_dict, stamp):
        float_arr = pred_dict_to_f32_multi_arr(pred_dict, stamp)
        self.det_publisher.publish(float_arr)

def RunInferenceServer(args, bar, period_sec, shr_tracker_time_sec):
    rclpy.init(args=None)
    node = InferenceServer(args, period_sec)
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            node.infer_loop(bar, shr_tracker_time_sec)
        prof.export_chrome_trace("trace.json")
    else:
        node.infer_loop(bar, shr_tracker_time_sec)

    node.destroy_node()
    rclpy.shutdown()

    bar.wait()

def RunStreamingEvaluator(args, bar, period_sec, shr_tracker_time_sec):
    rclpy.init(args=None)
    node = StreamingEvaluator(args, period_sec, shr_tracker_time_sec)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    node.start_sampling(bar)
    executor.spin()

    # Wont reach here

    #executor.shutdown()
    #node.destroy_node()
    #rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    cmdline_args = parser.parse_args()

    shr_tracker_time_sec = Value('d', 0.) # double
    bar = Barrier(2)
    period_sec = 0.05
    p2 = Process(target=RunStreamingEvaluator, args=(cmdline_args, bar, period_sec, shr_tracker_time_sec))
    p2.start()

    RunInferenceServer(cmdline_args, bar, period_sec, shr_tracker_time_sec)

    print('InferenceServer done')
    p2.terminate()
    p2.join()
    print('StreamingEvaluator done')

if __name__ == '__main__':
    main()

