import math
import os
import datetime
import torch
from std_msgs.msg import Header, MultiArrayDimension
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import TransformStamped, Twist, Point
from autoware_perception_msgs.msg import  DetectedObjects, DetectedObject, ObjectClassification
from valo_msgs.msg import Float32MultiArrayStamped
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from tf_transformations import quaternion_from_euler

from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

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
    ]
    point_step = 12

    if points.shape[1] > 3:
        point_cloud.fields.append(PointField(name='intensity', offset=12,
                datatype=PointField.FLOAT32, count=1))
        point_step += 4

    if points.shape[1] > 4:
        point_cloud.fields.append(PointField(name='time', offset=16,
            datatype=PointField.FLOAT32, count=1))
        point_step += 4

    point_cloud.point_step = point_step
    point_cloud.is_bigendian = False
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

def pred_dict_to_f32_multi_arr(pred_dict, stamp):
    pred_boxes  = pred_dict['pred_boxes'] # (N, 9) #xyz(3) dim(3) yaw(1) vel(2)
    if pred_boxes.size(0) > 0:
        pred_scores = pred_dict['pred_scores'].unsqueeze(-1) # (N, 1)
        pred_labels = pred_dict['pred_labels'].float().unsqueeze(-1) # (N, 1)
        all_data = torch.cat((pred_boxes, pred_scores, pred_labels), dim=1)
    else:
        all_data  = torch.empty((0, 11))

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

    return float_arr

def f32_multi_arr_to_pred_dict(float_arr):
    if len(float_arr.array.layout.dim) == 0: # empty det
        boxes, scores, labels = torch.empty((0, 9)), torch.empty(0), \
            torch.empty(0, dtype=torch.long)
    else:
        num_objs = float_arr.array.layout.dim[0].size;
        all_data = torch.tensor(float_arr.array.data, dtype=torch.float).view(num_objs, -1)
        boxes, scores, labels = all_data[:, :9], all_data[:, 9], all_data[:, 10].long()

    return {
        'pred_boxes': boxes,
        'pred_scores': scores,
        'pred_labels': labels
    }

def f32_multi_arr_to_detected_objs(float_arr, cls_mapping):
    SIGN_UNKNOWN=1
    BOUNDING_BOX=0

    # -1 car truck bus bicyle pedestrian

    all_objs = DetectedObjects()
    all_objs.header = float_arr.header

    num_objs = float_arr.array.layout.dim[0].size;
    all_data = torch.tensor(float_arr.array.data, dtype=torch.float).view(num_objs, -1)

    pred_boxes = all_data[:, :6]
    yaws = all_data[:, 6]
    vel_x = all_data[:, 7]
    vel_y = all_data[:, 8]
    pred_scores = all_data[:, 9]
    pred_labels = all_data[:, 10].long()

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

        q = quaternion_from_euler(0, 0, yaws[i])
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

    return all_objs

def get_dataset(cfg):
    pth = os.environ['PCDET_PATH']
    log_file = ('tools/tmp_results/log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    log_file = os.path.join(pth, log_file)
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    #log_config_to_file(cfg, logger=logger)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=1,
        dist=False, workers=0, logger=logger, training=False
    )

    return logger, test_set

def collate_dataset(indices, dataset):
    #return [dataset.collate_batch([dataset[idx]]) for idx in indices]
    dicts = [None] * len(indices)
    for idx, ind in enumerate(indices):
        data_dict = dataset.get_metadata_dict(ind)
        data_dict['points'] = dataset.get_lidar_with_sweeps(ind,
                max_sweeps=dataset.dataset_cfg.MAX_SWEEPS)
        dicts[idx] = dataset.collate_batch([data_dict])
    return dicts

def load_dataset_metadata(indices, dataset):
    return [dataset.get_metadata_dict(idx) for idx in indices]

def get_debug_pts(indices, dataset):
    return [points_to_pc2(dataset.get_lidar_with_sweeps(i)[:, :-1]) \
                for i in indices]

def get_gt_objects(indices, dataset, cls_mapping):
    objs = [None] * len(indices)
    dummy_stamp = TimeMsg() # will be overwritten later
    for idx, ind in enumerate(indices):
        gt_dict = dataset.get_gt_as_pred_dict(ind)
        if gt_dict['pred_boxes'].size(0) == 0:
            objs[idx] = None
        else:
            float_arr = pred_dict_to_f32_multi_arr(gt_dict, dummy_stamp)
            objs[idx] = f32_multi_arr_to_detected_objs(float_arr, cls_mapping)
    return objs

def get_debug_pts_and_gt_objects(indices, dataset, cls_mapping):
    debug_pts = get_debug_pts(indices, dataset)
    gt_objects = get_gt_objects(indices, dataset, cls_mapping)
    return debug_pts, gt_objects

def gen_viz_rectangle(vertices_xyz, rect_id):
    marker = Marker()

    marker.header.frame_id = "base_link"  # Replace with the appropriate frame
    #marker.header.stamp =  # set it later

    marker.ns = "rectangle"
    marker.id = rect_id
    marker.type = Marker.LINE_STRIP  # To draw a line connecting the vertices
    marker.action = Marker.ADD

    # Define the marker properties (scale, color, etc.)
    marker.scale.x = 0.13  # Line width
    marker.color.r = 1.0
    marker.color.g = 0.47
    marker.color.b = 0.0
    marker.color.a = 1.0  # Full opacity

    # Add points for the rectangle
    for vertex in vertices_xyz:
        point = Point()
        point.x, point.y, point.z = vertex
        marker.points.append(point)

    # Close the rectangle by adding the first point again
    marker.points.append(Point(x=vertices_xyz[0][0], y=vertices_xyz[0][1], z=vertices_xyz[0][2]))

    return marker


def gen_viz_filled_rectangle(vertices_xyz, rect_id, opacity=0.2):
    marker = Marker()

    marker.header.frame_id = "base_link"  # Replace with the appropriate frame
    #marker.header.stamp =  # set it later

    marker.ns = "filled_rectangle"
    marker.id = rect_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD

    marker.pose.position.x = (vertices_xyz[0][0] + vertices_xyz[1][0]) / 2.0
    marker.pose.position.y = (vertices_xyz[1][1] + vertices_xyz[2][1]) / 2.0
    marker.pose.position.z = -10.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = vertices_xyz[1][0] - vertices_xyz[0][0]
    marker.scale.y = vertices_xyz[1][1] - vertices_xyz[2][1]
    marker.scale.z = 0.01  # Thickness (small to make it look flat)

    marker.color.r = 1.0
    marker.color.g = 0.47
    marker.color.b = 0.0
    marker.color.a = opacity  # Opacity

    return marker


