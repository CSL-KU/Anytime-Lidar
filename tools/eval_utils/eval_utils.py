import pickle
import json
import time
import gc
import os
import copy

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from eval_utils.centerpoint_tracker import CenterpointTracker as Tracker

speed_test = False
visualize = False
visualize_ros2 = False
if visualize:
    import open3d
    from visual_utils import open3d_vis_utils as V

if visualize_ros2:
    import rclpy
    from rclpy.node import Node
    from autoware_auto_perception_msgs.msg import DetectedObjects, ObjectClassification
    from valo_msgs.msg import Float32MultiArrayStamped
    from sensor_msgs.msg import PointCloud2, PointField
    from visualization_msgs.msg import Marker, MarkerArray
    from rclpy.clock import ClockType, Clock
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from inference_ros2 import pred_dict_to_f32_multi_arr, f32_multi_arr_to_detected_objs, \
            points_to_pc2, gen_viz_rectangle, gen_viz_filled_rectangle

    class VisualizationNode(Node):
        def __init__(self, cls_names):
            super().__init__('valo_visualizer')
            qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10)
            self.system_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
            self.det_debug_publisher = self.create_publisher(DetectedObjects, 'detected_objects_debug',
                    qos_profile)
            self.ground_truth_publisher = self.create_publisher(DetectedObjects, 'ground_truth_objects',
                    qos_profile)
            self.pc_publisher = self.create_publisher(PointCloud2, 'point_cloud',
                    qos_profile)

            self.heatmap_pub = self.create_publisher(MarkerArray, 'heatmap', 10)

            oc = ObjectClassification()
            self.cls_mapping = { cls_names.index(name)+1: oc.__getattribute__(name.upper()) \
                    for name in cls_names }

            self.heatmap = None

        def pred_dict_to_detected_objs(self, pred_dict, tstamp=None):
            if tstamp is None:
                tstamp = self.system_clock.now().to_msg()
            float_arr = pred_dict_to_f32_multi_arr(pred_dict, tstamp)
            return f32_multi_arr_to_detected_objs(float_arr, self.cls_mapping)

        def publish_vis_data(self, dets_pred_dict, gt_pred_dict=None, points=None, tstamp=None, heatmap_ma=None):
            if tstamp is None:
                tstamp = self.system_clock.now().to_msg()

            if heatmap_ma is not None:
                self.heatmap_pub.publish(heatmap_ma)

            if points is not None:
                num_fields = points.shape[1]
                assert num_fields >= 3 and num_fields <= 5
                pc_msg = points_to_pc2(points)
                pc_msg.header.frame_id = 'base_link'
                pc_msg.header.stamp = tstamp
                self.pc_publisher.publish(pc_msg)

            det_objs = self.pred_dict_to_detected_objs(dets_pred_dict, tstamp)
            self.det_debug_publisher.publish(det_objs)

            if gt_pred_dict is not None:
                gt_objs = self.pred_dict_to_detected_objs(gt_pred_dict, tstamp)
                self.ground_truth_publisher.publish(gt_objs)

        def publish_frame(self, pc_range):
            vertices_xyz = np.array((
                (pc_range[0], pc_range[1], 0.), #-x -y
                (pc_range[3], pc_range[1], 0.), #+x -y
                (pc_range[3], pc_range[4], 0.), #+x +y
                (pc_range[0], pc_range[4], 0.)  #-x +y
            ))
            whole_area_rect = gen_viz_rectangle(vertices_xyz, 9999)
            whole_area_rect.header.stamp = self.system_clock.now().to_msg()

            ma = MarkerArray()
            ma.markers.append(whole_area_rect)
            self.heatmap_pub.publish(ma)



def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    # Forward once for initialization and calibration
    batch_size = dataloader.batch_size
    if 'calibrate' in dir(model):
        torch.cuda.cudart().cudaProfilerStop()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                model.calibrate()
        torch.cuda.cudart().cudaProfilerStart()

    global speed_test
    num_samples = 100 if speed_test and len(dataset) >= 10 else len(dataset)
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    gc.disable()

    if visualize:
        V.initialize_visualizer()

    if visualize_ros2:
        rclpy.init(args=None)
        vis_node = VisualizationNode(cfg.CLASS_NAMES)

    det_elapsed_musec = []
        #def get_ts(data_dict):
        #    return token_to_pose[data_dict['metadata']['token']]['timestamp']

    pred_tuples = [None] * (100 if speed_test else len(dataloader))
    for i in range(len(dataloader)):
        if speed_test and i == num_samples:
            break
        if getattr(args, 'infer_time', False):
            start_time = time.time()
        data_indexes = [i*batch_size+j for j in range(batch_size) \
                if i*batch_size+j < len(dataset)]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred_dicts, ret_dict = model(data_indexes)
        det_elapsed_musec.append(model.last_elapsed_time_musec)
        disp_dict = {}

#        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
#        allocated_MB, reserved_MB = allocated // (1024**2), reserved // (1024**2)
#        print(f'Allocated: {allocated_MB} MB, Reserved: {reserved_MB} MB')

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        batch_dict = model.latest_batch_dict
        if visualize:
            # Can infer which detections are projection from the scores
            # -x -y -z +x +y +z
            pd = batch_dict['final_box_dicts'][0]
            V.draw_scenes(
                points=batch_dict['points'][:, 1:], ref_boxes=pd['pred_boxes'],
                gt_boxes=batch_dict['gt_boxes'].cpu().flatten(0,1).numpy(),
                ref_scores=pd['pred_scores'], ref_labels=pd['pred_labels'],
                max_num_tiles=(model.tcount if hasattr(model, 'tcount') else None),
                pc_range=model.vfe.point_cloud_range.cpu().numpy(),
                nonempty_tile_coords=batch_dict.get('nonempty_tile_coords', None),
                tile_coords=batch_dict.get('chosen_tile_coords', None),
                clusters=batch_dict.get('clusters', None))

        if visualize_ros2: # 200 ms on jetson orin
            det_pred_dict = batch_dict['final_box_dicts'][0]
            gt_pred_dict = dataset.get_gt_as_pred_dict(data_indexes[0])

            points = batch_dict['points']

            pub_heatmap = False
            ma = None
            if pub_heatmap:
                ult_heatmap = None
                if 'ult_heatmap' in batch_dict:
                    ult_heatmap = batch_dict['ult_heatmap'].cpu()
                elif 'pred_dicts' in batch_dict and 'hm' in batch_dict['pred_dicts'][0]:
                    heatmaps = [pd['hm'] for pd in batch_dict['pred_dicts']]
                    all_heatmaps = torch.cat(heatmaps, dim=1)
                    ult_heatmap, _ = torch.max(all_heatmaps, 1, keepdim=True)
                    ult_heatmap = ult_heatmap.cpu()
                if ult_heatmap is not None:
                    hm_h, hm_w = ult_heatmap.shape[2:]
                    if vis_node.heatmap is None:
                        pc_range = model.vfe.point_cloud_range
                        pc_range = pc_range.cpu().numpy()
                        vis_node.publish_frame(pc_range)

                        # heatmap tile
                        field_h = pc_range[4] - pc_range[1]
                        field_w = pc_range[3] - pc_range[0]
                        hm_tile_h = field_h / hm_h
                        hm_tile_w = field_w / hm_w

                        tile_verts = np.array((
                            (pc_range[0],           pc_range[1], 0.), #-x -y
                            (pc_range[0]+hm_tile_w, pc_range[1], 0.), #+x -y
                            (pc_range[0]+hm_tile_w, pc_range[1]+hm_tile_h, 0.), #+x +y
                            (pc_range[0],           pc_range[1]+hm_tile_h, 0.)  #-x +y
                        ))

                        ma = MarkerArray()
                        for k in range(hm_h):
                            for j in range(hm_w):
                                tv = tile_verts + np.array((hm_tile_w*k, hm_tile_h*j, 0.))
                                tile = gen_viz_filled_rectangle(tv, k*hm_w+j,
                                        float(ult_heatmap[0, 0, j, k]))
                                ma.markers.append(tile)
                        vis_node.heatmap = ma
                    else:
                        ma = vis_node.heatmap
                        for k in range(hm_h):
                            for j in range(hm_w):
                                a = float(ult_heatmap[0, 0, j, k])
                                ma.markers[k*hm_w+j].color.a = 0.5 if a > 0.1 else 0.

            points = points[points[:,-1] == 0.] # only one scan
            points = points[:, 1:-1].cpu().contiguous().numpy()

            if det_pred_dict['pred_boxes'].size(0) > 0:
                vis_node.publish_vis_data(det_pred_dict, gt_pred_dict, points,
                        tstamp=None, heatmap_ma=ma)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        bd = {k:batch_dict[k] for k in ('frame_id', 'metadata')}
        pred_tuples[i] = (bd, pred_dicts)
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

        if i % 100 == 0:
            gc.collect() # don't invoke this all the time as it slows down

    op = final_output_dir if args.save_to_file else None
    for bd, pred_dicts in pred_tuples:
        annos = dataset.generate_prediction_dicts(
            bd, pred_dicts, class_names, output_path=op
        )
        det_annos += annos

    gc.enable()

    if visualize:
        V.destroy_visualizer()

    if 'post_eval' in dir(model):
        model.post_eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    model.print_time_stats()

    if cfg.LOCAL_RANK != 0:
        return {}

    if speed_test:
        model.dump_eval_dict(ret_dict)
        model.clear_stats()
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    do_eval = (int(os.getenv('DO_EVAL', 1)) == 1)

    do_tracking=False
    if do_eval:
        if dataset.dataset_cfg.DATASET != 'NuScenesDataset':
            result_str, result_dict = dataset.evaluation(
                det_annos, class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir,
            )
        else:
            nusc_annos = {}
            result_str, result_dict = dataset.evaluation(
                det_annos, class_names,
                eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                output_path=final_output_dir,
                nusc_annos_outp=nusc_annos,
                #det_elapsed_musec=det_elapsed_musec,
            )

            if do_tracking:
                ## NUSC TRACKING START
                tracker = Tracker(max_age=6, hungarian=False)
                predictions = nusc_annos['results']
                with open('frames/frames_meta.json', 'rb') as f:
                    frames=json.load(f)['frames']

                nusc_trk_annos = {
                    "results": {},
                    "meta": None,
                }
                size = len(frames)

                print("Begin Tracking\n")
                start = time.time()
                for i in range(size):
                    token = frames[i]['token']

                    # reset tracking after one video sequence
                    if frames[i]['first']:
                        # use this for sanity check to ensure your token order is correct
                        # print("reset ", i)
                        tracker.reset()
                        last_time_stamp = frames[i]['timestamp']

                    time_lag = (frames[i]['timestamp'] - last_time_stamp)
                    last_time_stamp = frames[i]['timestamp']

                    preds = predictions[token]

                    outputs = tracker.step_centertrack(preds, time_lag)
                    annos = []

                    for item in outputs:
                        if item['active'] == 0:
                            continue
                        nusc_anno = {
                            "sample_token": token,
                            "translation": item['translation'],
                            "size": item['size'],
                            "rotation": item['rotation'],
                            "velocity": item['velocity'],
                            "tracking_id": str(item['tracking_id']),
                            "tracking_name": item['detection_name'],
                            "tracking_score": item['detection_score'],
                        }
                        annos.append(nusc_anno)
                    nusc_trk_annos["results"].update({token: annos})
                end = time.time()
                second = (end-start)
                speed=size / second
                print("The speed is {} FPS".format(speed))
                nusc_trk_annos["meta"] = {
                    "use_camera": False,
                    "use_lidar": True,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False,
                }

                with open('tracking_result.json', "w") as f:
                    json.dump(nusc_trk_annos, f)

                #result is nusc_annos
                dataset.tracking_evaluation(
                    output_path=final_output_dir,
                    res_path='tracking_result.json'
                )

                ## NUSC TRACKING END

    if do_eval:
        logger.info(result_str)
        ret_dict.update(result_dict)
        ret_dict['result_str'] = result_str
    else:
        print('Skipping evaluation.')
        print('Dumping eval data')
        t0 = time.time()
        with open('eval.pkl', 'wb') as f:
            all_data = [dataset, det_annos, class_names,cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
                final_output_dir, {}, det_elapsed_musec]
            pickle.dump(all_data, f)
        print(f'Dumping took {(time.time() - t0):.2f} seconds.')

#    if cfg.MODEL.STREAMING_EVAL:
#        ret_dict['e2e_dl_musec'] = e2e_dl_musec

#    logger.info('Result is saved to %s' % result_dir)
#    logger.info('****************Evaluation done.*****************')

    model.dump_eval_dict(ret_dict)
    model.clear_stats()

    return ret_dict


if __name__ == '__main__':
    pass
