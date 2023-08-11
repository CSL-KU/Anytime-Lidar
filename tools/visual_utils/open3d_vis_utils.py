"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, max_num_tiles=18, pc_range=None, tile_coords=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    #vis.get_render_option().point_size = 1.0
    #vis.get_render_option().background_color = np.zeros(3)
    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        #pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        # Baby blue
        baby_blue = np.array([[137., 207., 240.]])/255.
        cornflower_blue = np.array([[100., 149., 237.]])/255.

        pts.colors = open3d.utility.Vector3dVector(np.repeat(cornflower_blue, points.shape[0], axis=0))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    if tile_coords is not None:
        #Draw all tiles, but change the color of chosen ones
        #pc range: -x -y -z +x +y +z
        tc = torch.from_numpy(tile_coords)
        tile_w = (pc_range[3] - pc_range[0]) / max_num_tiles
        tile_h = (pc_range[4] - pc_range[1])

        v_top = np.array((pc_range[0], pc_range[1]))
        v_bot = np.array((pc_range[0], pc_range[4]))
        vertices_top = np.array([v_top + (tile_w*i, 0) for i in range(max_num_tiles+1)])
        vertices_bot = np.array([v_bot + (tile_w*i, 0) for i in range(max_num_tiles+1)])
        vertices = np.concatenate((vertices_top, vertices_bot), axis=0)
        vertices_low = np.concatenate((vertices, np.full((vertices.shape[0],1), -3,
            dtype=vertices.dtype)), axis=1)
        vertices_high = vertices_low + (0,0,0.1)

        lines_chosen = []
        lines_other = []
        s = vertices_low.shape[0]//2
        for t in range(max_num_tiles):
            vl =[(t,t+1), (t,t+s), (t+s,t+s+1), (t+1,t+s+1)]

            if t in tile_coords:
                lines_chosen.extend(vl)
            else:
                lines_other.extend(vl)

        print('Tile coords:', tile_coords)

        all_vertices = (vertices_low, vertices_high)
        all_lines = (lines_other, lines_chosen)
        all_colors = (np.array((217, 136, 128))/255., np.array((125, 206, 160))/255.)
        for vertices, lines, colors in zip(all_vertices, all_lines, all_colors):
            o3d_vertices = open3d.utility.Vector3dVector(vertices)
            o3d_vertex_pairs = open3d.utility.Vector2iVector(np.array(lines))
            rectangles = open3d.geometry.LineSet(o3d_vertices, o3d_vertex_pairs)
            rectangles.paint_uniform_color(colors)
            vis.add_geometry(rectangles)
    else:
        # Assume there are 16 vertical tiles
        pass

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        #if ref_labels is None:
        #    line_set.paint_uniform_color(color)
        #else:
        #    line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        c = np.array((220, 118, 51))/255. if score[i] < 0.3 else np.array((39, 174, 96))/255.
        line_set.paint_uniform_color(c)
        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
