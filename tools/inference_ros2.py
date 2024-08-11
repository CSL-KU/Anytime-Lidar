import _init_path
import argparse
import datetime
import os
import time
import json
from pathlib import Path

import torch
import gc

from eval_utils import eval_utils
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
from autoware_auto_perception_msgs.msg import TrackedObjects, DetectedObjects, DetectedObject, ObjectClassification
from valo_msgs.msg import Float32MultiArrayStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from multiprocessing import Process, Barrier, Pool
import concurrent
import threading

VALO_DEBUG = False

def seconds_to_TimeMsg(seconds : float):
    sec_int = int(math.floor(seconds))
    return TimeMsg(sec=sec_int, nanosec=int((seconds-sec_int)*1e9))


def points_to_pc2(points):
    point_cloud = PointCloud2()
    #point_cloud.header = Header()
    #point_cloud.header.frame_id = 'base_link'
    # Assign stamp later
    point_cloud.height = 1
    point_cloud.width = points.shape[0]
    point_cloud.is_dense = True

    # Define the fields of the PointCloud2 message
    point_cloud.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    #    PointField(name='time', offset=16, datatype=PointField.FLOAT32, count=1),
    ]
    point_cloud.is_bigendian = False
    point_cloud.point_step = 16  # 4 fields x 4 bytes each
    point_cloud.row_step = point_cloud.point_step * points.shape[0]

    # Flatten the array for the data field
    point_cloud.data = points.tobytes()
    return point_cloud

def pose_to_tf(translation, rotation_q, stamp, frame_id, child_frame_id):
    t = TransformStamped()

    t.header.stamp = stamp
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id

    t.transform.translation.x = translation[0]
    t.transform.translation.y = translation[1]
    t.transform.translation.z = translation[2]

    t.transform.rotation.w = rotation_q[0]
    t.transform.rotation.x = rotation_q[1]
    t.transform.rotation.y = rotation_q[2]
    t.transform.rotation.z = rotation_q[3]

    return t

def get_dataset(cfg):
    log_file = ('./tmp_results/log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    #log_config_to_file(cfg, logger=logger)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )
    return logger, test_set

def collate_dataset(indices, dataset):
    #return [dataset.collate_batch([dataset[idx]]) for idx in indices]
    dicts = []
    for i in indices:
        data_dict = dataset.get_metadata_dict(i)
        data_dict['points'] = dataset.get_lidar_with_sweeps(i,
                max_sweeps=dataset.dataset_cfg.MAX_SWEEPS)
        dicts.append(dataset.collate_batch([data_dict]))
    return dicts


def load_dataset_metadata(indices, dataset):
    return [dataset.get_metadata_dict(idx) for idx in indices]

def get_debug_pts(indices, dataset):
    return [points_to_pc2(dataset.get_lidar_with_sweeps(i)[:, :-1]) \
                for i in indices]

class StreamingEvaluator(Node):
    def __init__(self, args, period_sec):
        super().__init__('streaming_evaluator')

        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        self.period_sec = period_sec

        # receive objects and update the buffer
        self.all_tracked_objects = [] #TrackedObjects()]
        self.all_detected_objects = []
        self.sampled_indices = []

        self.cb_group = rclpy.callback_groups.ReentrantCallbackGroup()
        #self.cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()

        self.cmd_sub = self.create_subscription(Header, 'evaluator_cmd',
                self.cmd_callback, 10) #, callback_group=self.cb_group)

        self.tracker_sub = self.create_subscription(TrackedObjects, 'tracked_objects',
                self.tracker_callback, 10, callback_group=self.cb_group)
        if VALO_DEBUG:
            self.detector_sub = self.create_subscription(DetectedObjects, 'detected_objects_debug',
                self.detector_callback, 10, callback_group=self.cb_group)

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

        self.dataset = test_set
        num_samples = len(self.dataset)
        print('[SE] Num dataset elems:', num_samples)

        nump=6
        self.debug_points = []
        metadata_arr = []
        splits = np.array_split(np.arange(num_samples), nump)
        with Pool(processes=nump) as pool:
            results = [pool.apply_async(get_debug_pts, (split.tolist(), self.dataset)) \
                    for split in splits]
            for res in results:
                self.debug_points += res.get()

        print('[SE] Loaded debug points')

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

    def tracker_callback(self, msg):
         # save the time of msg arrival for streaming eval
        duration = (self.system_clock.now() - self.time_to_sub).to_msg()
        msg.header.stamp = TimeMsg(sec=duration.sec, nanosec=duration.nanosec)
        self.all_tracked_objects.append(msg)

    def detector_callback(self, msg):
        duration = (self.system_clock.now() - self.time_to_sub).to_msg()
        msg.header.stamp = TimeMsg(sec=duration.sec, nanosec=duration.nanosec)
        self.all_detected_objects.append(msg)

    def start_sampling(self, bar):
        self.bar = bar
        self.bar.wait()
        self.init_time = time.time()

        self.pose_timer = self.create_timer(self.period_sec/2, self.pose_pub_timer_callback,
                callback_group=self.cb_group, clock=self.system_clock)

    # Receive buffer reset and start timer commands
    def cmd_callback(self, msg):
        cmd = msg.frame_id
        if cmd == 'stop_sampling': # stop timer and evaluate
            print(f'[SE] Sampled {len(self.all_detected_objects)} detected objects')
            print(f'[SE] Sampled {len(self.all_tracked_objects)} tracked objects')
            if VALO_DEBUG:
                sampled_objects = self.get_eval_samples(self.all_detected_objects)
            else:
                sampled_objects = self.get_streaming_eval_samples(self.all_tracked_objects)

            self.do_eval(sampled_objects)
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

        sampled_tracked_objects = []
        num_ds_elems = len(self.dataset)
        period_ns = int(self.period_sec * 1e9)
        for i in range(num_ds_elems):
            sample_time_ns = init_ns + i*period_ns
            aft = times_ns > sample_time_ns
            if aft[0] == True:
                sampled_tracked_objects.append(TrackedObjects())
                print('0', end=' ')
            elif aft[-1] == False:
                sampled_tracked_objects.append(all_objects[-1])
                print('-1', end=' ')
            else:
                sample_idx = np.argmax(aft) - 1
                sampled_tracked_objects.append(all_objects[sample_idx])
                print(sample_idx, end=' ')
        print()

        num_sampled = len(sampled_tracked_objects)
        return sampled_tracked_objects

    def get_eval_samples(self,  all_objects):
        num_ds_elems = len(self.dataset)
        while len(all_objects) < num_ds_elems:
            all_objects.append(all_objects[-1])
        return all_objects[:num_ds_elems]

    def do_eval(self, sampled_objects):
        #Convert them to openpcdet format
        inv_cls_mapping = [-1, 1, 2, 3, -1, -1, 4, 5]

        det_annos = []
        num_ds_elems = len(self.dataset)
        for i in range(num_ds_elems):
            data_dict = self.dataset.get_metadata_dict(i)
            for k, v in data_dict.items():
                data_dict[k] = [v] # make it a batch dict
            objs = sampled_objects[i]
            num_objs = len(objs.objects)
            boxes, scores, labels = torch.empty((num_objs, 9)), torch.empty(num_objs), torch.empty(num_objs, dtype=torch.long)

            mask = torch.ones(num_objs, dtype=torch.bool)
            for j, obj in enumerate(objs.objects):
                scores[j] = obj.existence_probability
                labels[j] = inv_cls_mapping[obj.classification[0].label]
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

            det_annos += self.dataset.generate_prediction_dicts(
                data_dict, data_dict['final_box_dicts'], self.dataset.class_names, output_path=None
            )

        fi = sampled_objects[0].header.frame_id
         
        nusc_annos = {} # not needed but keep it anyway
        result_str, result_dict = self.dataset.evaluation(
            det_annos, self.dataset.class_names,
            eval_metric=self.cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            output_path='./tmp_results',
            nusc_annos_outp=nusc_annos,
            boxes_in_global_coords=(fi != 'base_link'),
            #det_elapsed_musec=det_elapsed_musec,
        )

        print(result_str)

class InferenceServer(Node):
    def __init__(self, args, period_sec):
        super().__init__('inference_server')
        #self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, False)])
        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)

        self.init_model(args)

        self.period_sec = period_sec

        self.det_publisher = self.create_publisher(Float32MultiArrayStamped, 'detected_objects_raw', 1)

        self.cmd_publisher = self.create_publisher(Header, 'evaluator_cmd', 10)

        self.pc_idx_publisher = self.create_publisher(Header, 'point_cloud_idx', 10)
        if VALO_DEBUG:
            self.det_debug_publisher = self.create_publisher(DetectedObjects, 'detected_objects_debug', 10)

    def init_model(self, args):
        cfg_from_yaml_file(args.cfg_file, cfg)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)

        logger, test_set = get_dataset(cfg)

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.eval()
        model.cuda()

        self.dataset = model.dataset
        self.num_samples = len(self.dataset)
        self.batch_dicts_arr = [None] * self.num_samples

        tkn = model.token_to_scene[self.dataset.infos[0]['token']]
        self.scene_start_indices = [(tkn, 0)]
        for i in range(1, self.num_samples):
            new_tkn = model.token_to_scene[self.dataset.infos[i]['token']]
            if new_tkn != tkn:
                self.scene_start_indices.append((new_tkn, i))
            tkn = new_tkn
        self.scene_start_indices.append(('', self.num_samples))
        print('[IS] Scene start indices:', self.scene_start_indices)

        print('[IS] Calibrating...')
        torch.cuda.cudart().cudaProfilerStop()
        with torch.no_grad():
            model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

        # Remove the hooks now
        model.pre_hook_handle.remove()
        model.post_hook_handle.remove()

        self.model = model

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
                    print('Cleaning samples ', prev_ind, '...', ind-1)
                    for d in range(prev_ind, ind):
                        self.batch_dicts_arr[d] = None

                return

    def infer_loop(self, bar):
        model = self.model
        tdiffs = [0. for i in range(len(self.dataset))]

        # Disable garbage collection to make things deterministic
        # pytorchs cuda allocator should cleanup its own memory seperately from gc
        # NOTE I hope dram won't overflow!
        # 1G prealloc
        gc.disable()
        dummy_tensor = torch.empty(1024*1024*1024, device='cuda')
        torch.cuda.synchronize()
        del dummy_tensor

        gc.collect()
        bar.wait() # sync with timer

        init_time = time.time()
        print('[IS] Starting inference at', init_time)
        last_i, i, last_scene_token = -1, -1, ''
        #wakeup_time = init_time + self.period_sec
        while rclpy.ok():
            if VALO_DEBUG:
                i += 1
            else:
                i = int((time.time() - init_time) / self.period_sec)
                if i == last_i:
                    print(f'[IS] Trying to process the sample {i} again, skipping to {i+1}.')
                    i += 1
            if i >= self.num_samples:
                break
            last_i = i

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

            last_scene_token = scene_token
            batch_dict = self.batch_dicts_arr[i]
            self.batch_dicts_arr[i] = None
            model.measure_time_start('End-to-end')
            model.measure_time_start('PreProcess')
            start_time = time.time()

            # Tell Evaluator to publish point cloud for debug
            stamp_time = init_time + self.period_sec * i
            cur_stamp = seconds_to_TimeMsg(stamp_time)
            idx_msg = Header()
            idx_msg.frame_id = str(i)
            idx_msg.stamp = cur_stamp
            self.pc_idx_publisher.publish(idx_msg)

            deadline_sec_override, reset = model.initialize(sample_token)
            if reset:
                #Clear buffers
                model.latest_batch_dict = None

            with torch.no_grad():
                load_data_to_gpu(batch_dict)

                batch_dict['scene_reset'] = reset
                batch_dict['start_time_sec'] = start_time
                batch_dict['deadline_sec'] = float(cfg.MODEL.DEADLINE_SEC)
                batch_dict['abs_deadline_sec'] = start_time + batch_dict['deadline_sec']
                model.measure_time_end('PreProcess')

                batch_dict = model.forward(batch_dict)

                model.measure_time_start('PostProcess')
                if 'final_box_dicts' in  batch_dict:
                    if 'pred_ious' in batch_dict['final_box_dicts'][0]:
                        del batch_dict['final_box_dicts'][0]['pred_ious']
                    for k,v in batch_dict['final_box_dicts'][0].items():
                        batch_dict['final_box_dicts'][0][k] = v.cpu()

            model.latest_batch_dict = {k: batch_dict[k] for k in \
                    ['final_box_dicts', 'metadata', 'chosen_tile_coords']}

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

        eval_cmd = Header()
        eval_cmd.frame_id = 'stop_sampling'
        self.cmd_publisher.publish(eval_cmd)

        gc.enable()
        for i, tdiff in enumerate(tdiffs):
            if tdiff > 0:
                print(f'Deadline {i} missed with {tdiff * 1000} ms')
        model.print_time_stats()

    # This func takes less than 1 ms, ~0.6 ms
    def publish_dets(self, pred_dict, stamp):
        pred_boxes  = pred_dict['pred_boxes'] # (N, 9) #xyz(3) dim(3) yaw(1) vel(2)
        pred_scores = pred_dict['pred_scores'].unsqueeze(-1) # (N, 1)
        pred_labels = pred_dict['pred_labels'].float().unsqueeze(-1) # (N, 1)

        all_data = torch.cat((pred_boxes, pred_scores, pred_labels), dim=1)

        float_arr = Float32MultiArrayStamped()
        float_arr.header.frame_id = 'base_link'
        float_arr.header.stamp = stamp

        dim2 = MultiArrayDimension()
        dim2.label = "obj_attributes"
        dim2.size = all_data.shape[1]
        dim2.stride = all_data.shape[1]

        dim1 = MultiArrayDimension()
        dim1.label = "num_objects"
        dim1.size = all_data.shape[0]
        dim1.stride = all_data.shape[0] * all_data.shape[1]

        float_arr.array.layout.dim.append(dim1)
        float_arr.array.layout.dim.append(dim2)
        float_arr.array.layout.data_offset = 0
        float_arr.array.data = all_data.flatten().tolist()

        self.det_publisher.publish(float_arr)

        if VALO_DEBUG:
            self.publish_debug_detections(pred_dict, stamp)

    def publish_debug_detections(self, pred_dict, stamp):
        #NOTE not sure if the values on the left are correct
        SIGN_UNKNOWN=1
        BOUNDING_BOX=0

        # -1 car truck bus bicyle pedestrian
        cls_mapping = [-1, 1, 2, 3, 6, 7]

        all_objs = DetectedObjects()
        all_objs.header = Header()
        all_objs.header.stamp = stamp
        all_objs.header.frame_id = 'base_link'

        pred_boxes  = pred_dict['pred_boxes'] # (N, 9) #xyz(3) dim(3) yaw(1) vel(2)
        pred_labels = pred_dict['pred_labels'] # (N)
        pred_scores = pred_dict['pred_scores'] # (N)

        yaws = pred_boxes[:, 6]
        vel_x = pred_boxes[:, 7]
        vel_y = pred_boxes[:, 8]

        #yaws = -yaws -math.pi / 2 # to ros2 format, not sure if I need it
        quaterns = [quaternion_from_euler(0, 0, yaw) for yaw in yaws]

        linear_x = torch.sqrt(torch.pow(vel_x, 2) + torch.pow(vel_y, 2)).tolist()
        angular_z = (2 * (torch.atan2(vel_y, vel_x) - yaws)).tolist()

        for i in range(pred_labels.size(0)):
            obj = DetectedObject()
            obj.existence_probability = pred_scores[i].item()

            oc = ObjectClassification()
            oc.probability = 1.0;
            oc.label = cls_mapping[pred_labels[i].item()]

            obj.classification.append(oc)

            if oc.label <= 3: #it is an car-like object
                obj.kinematics.orientation_availability=SIGN_UNKNOWN

            pbox = pred_boxes[i].tolist()

            obj.kinematics.pose_with_covariance.pose.position.x = pbox[0]
            obj.kinematics.pose_with_covariance.pose.position.y = pbox[1]
            obj.kinematics.pose_with_covariance.pose.position.z = pbox[2]

            q = quaterns[i]
            obj.kinematics.pose_with_covariance.pose.orientation.x = q[0]
            obj.kinematics.pose_with_covariance.pose.orientation.y = q[1]
            obj.kinematics.pose_with_covariance.pose.orientation.z = q[2]
            obj.kinematics.pose_with_covariance.pose.orientation.w = q[3]

            obj.shape.type = BOUNDING_BOX
            obj.shape.dimensions.x = pbox[3]
            obj.shape.dimensions.y = pbox[4]
            obj.shape.dimensions.z = pbox[5]

            twist = Twist()
            twist.linear.x = linear_x[i]
            twist.angular.z = angular_z[i]
            obj.kinematics.twist_with_covariance.twist = twist
            obj.kinematics.has_twist = True

            all_objs.objects.append(obj)

        self.det_debug_publisher.publish(all_objs)

def RunInferenceServer(args, bar, period_sec):
    rclpy.init(args=None)
    node = InferenceServer(args, period_sec)
    
    node.infer_loop(bar)

    node.destroy_node()
    rclpy.shutdown()

    bar.wait()

def RunStreamingEvaluator(args, bar, period_sec):
    rclpy.init(args=None)
    node = StreamingEvaluator(args, period_sec)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    node.start_sampling(bar)
    executor.spin()

    # Won't reach here

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

    bar = Barrier(2)
    period_sec = 0.05
    p1 = Process(target=RunInferenceServer, args=(cmdline_args, bar, period_sec))
    p2 = Process(target=RunStreamingEvaluator, args=(cmdline_args, bar, period_sec))

    p1.start()
    p2.start()

    p1.join()
    print('InferenceServer done')
    p2.terminate()
    p2.join()
    print('StreamingEvaluator done')

if __name__ == '__main__':
    main()

#rate = self.create_rate(1.0/self.period_sec, self.get_clock())
## Express the boxes in world coordinates for tracker to work correctly
#cur_pose = model.token_to_pose[latest_token]
#pred_boxes = batch_dict['final_box_dicts'][0]['pred_boxes']
#pred_boxes = cuda_projection.move_to_world_coords(pred_boxes, cur_pose)
#batch_dict['final_box_dicts'][0]['pred_boxes'] = pred_boxes


