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
from strinf_ros2 import get_dataset, pred_dict_to_f32_multi_arr, f32_multi_arr_to_detected_objs

#VALO_DEBUG = False
#DO_DYN_SCHED = True
#PERIODIC_SCHED = False
#ANYTIME_CAPABLE = True
PROFILE = False

#assert (DO_DYN_SCHED != ALWAYS_BLOCK_SCHED)

# has to be converted to [N, 6]
def f32_multi_arr_to_tensor(float_arr):
    num_points = float_arr.array.layout.dim[0].size;
    num_attr = float_arr.array.layout.dim[1].size;
    np_arr = np.array(float_arr.array.data, dtype=np.float32)
    np_arr = np_arr.reshape((num_points,num_attr))
    points = torch.from_numpy(np_arr)
    return points

class InferenceNode(Node):
    def __init__(self, args, period_sec):
        super().__init__('inference_server')
        self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.init_model(args)

        self.period_sec = period_sec
        point_cloud_topic = '/dnn_inp_float_arr'
        output_topic = '/valor_detected_objs_as_arr'
        self.arr_publisher = self.create_publisher(Float32MultiArrayStamped, output_topic, 10)
        output_topic = '/valor_detected_objs'
        self.det_publisher = self.create_publisher(DetectedObjects, output_topic, 10)
        self.pc_sub = self.create_subscription(Float32MultiArrayStamped, point_cloud_topic,
                                               self.pc_callback, 10)
        self.sample_counter = 0

        # Define the transformation matrix from
        # velodyne_top frame to base_link frame
        self.vt_to_bl_tf = torch.tensor([
			[  0.032, -0.999,  0.015,  0.901],
			[  0.999,  0.032,  0.000,  0.000],
			[ -0.001,  0.015,  1.000,  2.066],
			[  0.000,  0.000,  0.000,  1.000]
        ], dtype=torch.float32)

    def tranform_to_base_link(self, objects):
        # Extract translation and orientation components
        translations = objects[:, :3]  # [x, y, z]

        # Apply transformation to translations
        translations_h = torch.cat([translations, torch.ones(translations.shape[0], 1)], dim=1)
        transformed_translations_h = torch.mm(self.vt_to_bl_tf, translations_h.T)
        objects[:, :3] = transformed_translations_h[:3, :].T  # Extract x, y, z

        # Update yaw based on rotation
        yaw_rotation_angle = torch.atan2(self.vt_to_bl_tf[1, 0], self.vt_to_bl_tf[0, 0])  # Extract yaw rotation
        objects[:, 6] += yaw_rotation_angle

        return objects

    def init_model(self, args):
        cfg_from_yaml_file(args.cfg_file, cfg)
        if args.set_cfgs is not None:
            cfg_from_list(args.set_cfgs, cfg)

        logger, test_set = get_dataset(cfg)

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
        model.eval()
        model.cuda()
        self.model_cfg = cfg

        self.dataset = model.dataset

        oc = ObjectClassification()
        cls_names = cfg.CLASS_NAMES
        self.cls_mapping = { cls_names.index(name)+1: oc.__getattribute__(name.upper()) \
                for name in cls_names }

        print('[IS] Calibrating...')
        torch.cuda.cudart().cudaProfilerStop()
        with torch.no_grad():
            model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

        # Remove the hooks now
        model.pre_hook_handle.remove()
        model.post_hook_handle.remove()
        model.latest_batch_dict = None
        model.last_elapsed_time_musec = 100000
        self.model = model

        dummy_tensor = torch.empty(1024**3, device='cuda')
        torch.cuda.synchronize()
        del dummy_tensor

    def pc_callback(self, multi_arr):
#        if len(multi_arr.array.data) == 0:
#            print('Empty input array!')
#            return

        with record_function("inference"):
            model = self.model
            start_time = time.time()
            model.measure_time_start('End-to-end')
            model.measure_time_start('PreProcess')
            #deadline_sec_override, reset = model.initialize(sample_token)

            with torch.no_grad():
                model.res_idx = 3

                tensor = f32_multi_arr_to_tensor(multi_arr).cuda()
                # the reference frame of awsim is baselink, so we need to change that to
                # lidar by decreasing z
                batch_id = torch.zeros(tensor.size(0), dtype=tensor.dtype,  device=tensor.device)

                batch_dict = {
                    'points': torch.cat((batch_id.unsqueeze(-1), tensor), dim=1)
                }
                torch.cuda.synchronize()

                #load_data_to_gpu(batch_dict)
                pts = batch_dict['points']
                #for i in range(6):
                #    print(i, torch.min(pts[:, i]), torch.max(pts[:, i]))

                batch_dict['batch_size'] = 1
                batch_dict['scene_reset'] = False
                batch_dict['start_time_sec'] = start_time
                batch_dict['deadline_sec'] = 10.0
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
                    ['final_box_dicts']}

            torch.cuda.synchronize()
            model.measure_time_end('PostProcess')
            model.measure_time_end('End-to-end')

            model.calc_elapsed_times() 
            model.last_elapsed_time_musec = int(model._time_dict['End-to-end'][-1] * 1000)

            pred_dict = batch_dict['final_box_dicts'][0]
            pred_dict['pred_boxes'] = self.tranform_to_base_link(pred_dict['pred_boxes'])

            self.publish_dets(pred_dict, multi_arr.header.stamp)
            finish_time = time.time()
            #print(batch_dict['final_box_dicts'][0]['pred_labels'].size(), (finish_time -start_time)*1e3, 'ms')
        self.sample_counter += 1
            #finishovski_time = time.time()

        if self.sample_counter % 100 == 0:
            model.print_time_stats()
            model.clear_stats()

    # This func takes less than 1 ms, ~0.6 ms
    def publish_dets(self, pred_dict, stamp):
        if pred_dict['pred_labels'].size(0) > 0:
            float_arr = pred_dict_to_f32_multi_arr(pred_dict, stamp)
            self.arr_publisher.publish(float_arr)
            det_objs = f32_multi_arr_to_detected_objs(float_arr, self.cls_mapping)
            self.det_publisher.publish(det_objs)
        else:
            print('Not publishing since no objects were detected...')

def RunInferenceNode(args, period_sec):
    rclpy.init(args=None)
    node = InferenceNode(args, period_sec)
    if PROFILE:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            try:
                rclpy.spin(node)
            except:
                pass
        prof.export_chrome_trace("trace.json")

    else:
        rclpy.spin(node)

    node.destroy_node()
    #rclpy.shutdown()

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    cmdline_args = parser.parse_args()
    period_sec = 0.1 # point cloud period
    RunInferenceNode(cmdline_args, period_sec)

    print('InferenceNode finished execution.')

if __name__ == '__main__':
    main()

