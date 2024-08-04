import _init_path
import argparse
import datetime
import os
import time
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
from std_msgs.msg import String, Header
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from autoware_auto_perception_msgs.msg import TrackedObjects, DetectedObjects, DetectedObject, ObjectClassification
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
import tf2_ros
import math
import threading
import tf_transformations
from copy import deepcopy
import numpy as np

def points_to_pc2(points):
    point_cloud = PointCloud2()
    point_cloud.header = Header()
    point_cloud.header.frame_id = 'base_link'
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
    point_cloud.point_step = 16  # 5 fields x 4 bytes each
    point_cloud.row_step = point_cloud.point_step * points.shape[0]

    # Flatten the array for the data field
    point_cloud.data = points.tobytes()
    return point_cloud


class InferenceServer(Node):

    def __init__(self):
        super().__init__('inference_server')
        self.init_model()

        self.mutex = threading.Lock()  # Create a mutex
        # this has two transforms each having 7 elems, x y z quaternion
        # cst csr ept epr
        self.cur_pose = torch.zeros(14, dtype=torch.float, device='cpu')

        self.re_callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        self.inf_subscription = self.create_subscription(String, 'inference_request',
                self.infer_callback, 1, callback_group=self.re_callback_group)
        self.inf_publisher = self.create_publisher(String, 'inference_request', 1,
                callback_group=self.re_callback_group)

        self.det_publisher = self.create_publisher(DetectedObjects, 'detected_objects', 10)

        self.detpub_helper_subs =  self.create_subscription(Header, 'publish_request',
                self.publish_dets_callback, 10, callback_group=self.re_callback_group)
        self.publish_requester = self.create_publisher(Header, 'publish_request', 10)

        # receive and update the buffer
        self.tracker_mutex = threading.Lock()  # Create a mutex
        self.latest_tracked_objs = TrackedObjects()
        self.all_tracked_objects = []
        self.tracker_sub = self.create_subscription(TrackedObjects, 'tracked_objects',
                self.tracker_callback, 10, callback_group=self.re_callback_group)

        # Broadcast every 25 ms
        period_sec = 0.025
        self.br = tf2_ros.TransformBroadcaster(self)
        self.tf_timer = self.create_timer(period_sec, self.tf_timer_callback,
                callback_group=self.re_callback_group)

        # For debug
        qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10)
        self.pc_publisher = self.create_publisher(PointCloud2, 'point_cloud', qos_profile)

        gc.disable()

    def init_model(self):
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
        parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
        parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                            help='set extra config keys if needed')

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)

        log_file = ('./tmp_results/log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
        log_config_to_file(cfg, logger=logger)

        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
            dist=False, workers=0, logger=logger, training=False
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.eval()
        model.cuda()

        # Load all data before execution
        dataset = model.dataset
        print(f'Loading dataset to memory, num samples: {len(dataset)}')
        self.batch_dicts_arr = [dataset.collate_batch([dataset[i]]) for i in range(len(dataset))]

        self.debug_pts_arr = []
        for bd in self.batch_dicts_arr:
            pts = bd['points'][:, 1:] # remove batch id
            mask = (pts[:, -1] == .0)
            self.debug_pts_arr.append(points_to_pc2(pts[mask][:, :-1]))

        print('Calibrating')
        torch.cuda.cudart().cudaProfilerStop()
        with torch.no_grad():
            model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

        # Remove the hooks now
        model.pre_hook_handle.remove()
        model.post_hook_handle.remove()

        # Disable garbage collection to make things deterministic
        # pytorchs cuda allocator should cleanup its own memory seperately from gc
        # NOTE I hope dram won't overflow!

        # 1G prealloc
        dummy_tensor = torch.empty(1024*1024*1024, device='cuda')
        torch.cuda.synchronize()
        del dummy_tensor

        self.model = model

    def pose_to_tf(self, translation, rotation_q, stamp, frame_id, child_frame_id):
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

    def tf_timer_callback(self):
        with self.mutex:
            pose = self.cur_pose.tolist()
            stamp = deepcopy(self.cur_stamp)

        tf_ep = self.pose_to_tf(pose[7:10], pose[10:], stamp, 'world', 'body')
        self.br.sendTransform(tf_ep)
        tf_cs = self.pose_to_tf(pose[:3], pose[3:7], stamp, 'body', 'base_link')
        self.br.sendTransform(tf_cs)

    def publish_detections(self, pred_dict, stamp):
        tstart = time.time()
        #NOTE not sure if the values on the left are correct
        SIGN_UNKNOWN=1
        BOUNDING_BOX=0
        cls_mapping = {
                1:1, # car
                2:2, # truck
                3:3, # bus
                4:6, # bicycle
                5:7, # pedestrian
        }

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
        quaterns = [tf_transformations.quaternion_from_euler(0, 0, yaw) for yaw in yaws]

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

        self.det_publisher.publish(all_objs)
        tend = time.time()
        #print(f'Publishing {len(all_objs.objects)} objects, it took {tend-tstart} secs')
        #print(f'Publishing {len(all_objs.objects)} objects at time {round(tend,3)}')

    def publish_dets_callback(self, msg):
        idx = int(msg.frame_id)
        pred_dict = self.batch_dicts_arr[idx]['final_box_dicts'][0]
        self.publish_detections(pred_dict, msg.stamp)

        # Also publish point cloud for debug
        # NOTE these are filtered points!
        #points = self.batch_dicts_arr[idx]['points_np']

        pc2_msg = self.debug_pts_arr[idx]
        pc2_msg.header.stamp = msg.stamp
        self.pc_publisher.publish(pc2_msg)

    def tracker_callback(self, msg):
        with self.tracker_mutex:
            self.latest_tracked_objs = msg
        #print(f'Received {len(msg.objects)} tracked objects at time {round(time.time(),3)}')

    def eval_callback(self):
        with self.tracker_mutex:
            latest_tobj = deepcopy(self.latest_tracked_objs)
        self.all_tracked_objects.append(latest_tobj)

    def infer_callback(self, msg):
        model = self.model
        tdiffs = [0. for i in range(len(self.batch_dicts_arr))]
        print('Starting inference')

        # streaming evaluation sampler
        pc_period_sec= 0.100
        self.eval_timer = self.create_timer(pc_period_sec, self.eval_callback,
                callback_group=self.re_callback_group)
        #time.sleep(0.1)

        rate = self.create_rate(10)  # 10 Hz rate
        init_time = time.time()
        #for i, batch_dict in enumerate(self.batch_dicts_arr):
        last_processed_idx = -1
        while True:
            i = int((time.time() - init_time) / pc_period_sec)
            if i >= len(self.batch_dicts_arr):
                break
            if i == last_processed_idx:
                print('Trying to process the same sample, skipping to next.')
                i += 1
            batch_dict = self.batch_dicts_arr[i]
            last_processed_idx = i

            model.measure_time_start('End-to-end')
            model.measure_time_start('PreProcess')
            start_time = time.time()

            latest_token = batch_dict['metadata'][0]['token']
            with self.mutex:
                self.cur_pose = model.token_to_pose[latest_token]
                self.cur_stamp = self.get_clock().now().to_msg()
            deadline_sec_override, reset = model.initialize(latest_token)
            if reset:
                #Clear buffers
                model.latest_batch_dict = None
                with self.tracker_mutex:
                    self.latest_tracked_objs = TrackedObjects()

            with torch.no_grad():
                load_data_to_gpu(batch_dict)

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
           
            torch.cuda.synchronize()
            model.measure_time_end('PostProcess')
            model.measure_time_end('End-to-end')
            finish_time = time.time()

            #Following stuff takes 0.4 ms
            model.calc_elapsed_times() 
            model.last_elapsed_time_musec = int(model._time_dict['End-to-end'][-1] * 1000)
            model.latest_batch_dict = batch_dict

            tdiffs[i] = round(finish_time - batch_dict['abs_deadline_sec'], 3)

            # keep final_box_dicts
            to_keep = ('final_box_dicts', 'frame_id', 'metadata')
            batch_dict = {k:batch_dict[k] for k in to_keep}

            self.batch_dicts_arr[i] = batch_dict
            req = Header()
            req.frame_id = str(i)
            req.stamp = deepcopy(self.cur_stamp)
            self.publish_requester.publish(req)

            rate.sleep()
            #publish_time = time.time()
            #pit = round((publish_time - finish_time) * 1000, 2)
            #print(f'Post infer took {pit} ms')

            #if i % 10 == 0:
            #    print(torch.cuda.memory_summary())
        self.eval_timer.cancel()

        gc.enable()
        for i, tdiff in enumerate(tdiffs):
            if tdiff > 0:
                print(f'Deadline {i} missed with {tdiff * 1000} ms')
        model.print_time_stats()

        #for tracked_objs in self.all_tracked_objects:
        #    print('Tracked objects:', len(tracked_objs.objects))
        print(f'Sampled {len(self.all_tracked_objects)} tracker results')

        #convert these back to batch dict format

#        det_annos = []
#        for bd in batch_dicts_arr:
#            det_annos += dataset.generate_prediction_dicts(
#                bd, bd['final_box_dicts'], dataset.class_names, output_path=None
#            )
#         
#        nusc_annos = {} # not needed but keep it anyway
#        result_str, result_dict = dataset.evaluation(
#            det_annos, dataset.class_names,
#            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
#            output_path='./tmp_results',
#            nusc_annos_outp=nusc_annos,
#            boxes_in_global_coords=True,
#            #det_elapsed_musec=det_elapsed_musec,
#        )
#
#        print(result_str)

def main(args=None):
    rclpy.init(args=args)

    node = InferenceServer()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    s = String()
    s.data = "Bismillahirrahmanirrahim"
    node.inf_publisher.publish(s)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


## Express the boxes in world coordinates for tracker to work correctly
#cur_pose = model.token_to_pose[latest_token]
#pred_boxes = batch_dict['final_box_dicts'][0]['pred_boxes']
#pred_boxes = cuda_projection.move_to_world_coords(pred_boxes, cur_pose)
#batch_dict['final_box_dicts'][0]['pred_boxes'] = pred_boxes


