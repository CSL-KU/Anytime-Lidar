import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

def visualize_bev_detections(
    detections: np.ndarray,
    ground_truths: np.ndarray,
    save_path: Optional[str] = None,
    range_limit: float = 50.0,
    detection_color: str = 'red',
    gt_color: str = 'green',
    figsize: Tuple[int, int] = (10, 10),
    swap_wh = False,
) -> None:
    """
    Visualize Bird's Eye View (BEV) object detections in sensor frame.
    
    Args:
        detections: Numpy array of shape (N, 7) with format [x, y, z, width, length, height, yaw]
        ground_truths: Numpy array of shape (M, 7) with format [x, y, z, width, length, height, yaw]
        save_path: Optional path to save the visualization
        range_limit: Visualization range limit in meters
        detection_color: Color for detection boxes
        gt_color: Color for ground truth boxes
        figsize: Figure size for the plot
        
    Returns:
        None
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set equal aspect ratio and limits
    ax.set_xlim(-range_limit, range_limit)
    ax.set_ylim(-range_limit, range_limit)
    ax.set_aspect('equal')
    
    # Setup grid and labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('Bird\'s Eye View Object Detection (Sensor Frame)', fontsize=14)
    
    # Draw sensor origin (LiDAR position)
    ax.scatter(0, 0, color='blue', s=100, label='Sensor Origin')
    
    # Draw orientation axes
    ax.arrow(0, 0, 3, 0, head_width=0.7, head_length=1.0, fc='blue', ec='blue', alpha=0.8)
    ax.arrow(0, 0, 0, 3, head_width=0.7, head_length=1.0, fc='green', ec='green', alpha=0.8)
    ax.text(3.5, 0, 'X', color='blue', fontsize=12)
    ax.text(0, 3.5, 'Y', color='green', fontsize=12)
    
    # Draw range rings
    for r in [10, 20, 30, 40]:
        circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.5)
        ax.add_artist(circle)
        ax.text(r, 0, f"{r}m", color='gray', ha='left', va='center', alpha=0.7)

    if swap_wh:
        tmp = detections[:, 3].copy()
        detections[:, 3] = detections[:, 4]
        detections[:, 4] = tmp
        tmp = ground_truths[:, 3].copy()
        ground_truths[:, 3] = ground_truths[:, 4]
        ground_truths[:, 4] = tmp
    
    # Draw detection boxes
    for det in detections:
        Box(det[:3], det[3:6], Quaternion(axis=[0, 0, 1],
            radians=det[6])).render(ax, colors=('r', 'r', 'r'))

    for gt in ground_truths:
        Box(gt[:3], gt[3:6], Quaternion(axis=[0, 0, 1],
            radians=gt[6])).render(ax, colors=('g', 'g', 'g'))

    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()
    
