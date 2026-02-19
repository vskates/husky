from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory("grasp_inference_pkg")

    model_path = os.path.join(pkg_share, "models", "model_5700.pth")
    seg_model_path = os.path.join(pkg_share, "models", "yolo_finetuned.pt")  # положите сюда веса (или укажите свой путь)

    return LaunchDescription([
        Node(
            package="grasp_inference_pkg",
            executable="grasp_node",
            name="heightmap_node",
            output="screen",
            parameters=[{
                "pcd_topic": "/camera/camera/depth/color/points",
                "target_frame": "camera_link",
                "hm_size": 224,
                "hm_resolution": 0.002,
                "plane_min": [-0.224, -0.224],
                "plane_max": [0.224, 0.224],
                "out_prefix": "heightmap",

                # YOLOv8-seg
                "seg_model_path": seg_model_path,
                "seg_imgsz": 640,
                "seg_conf": 0.25,
                "seg_iou": 0.7,
                "seg_force_cpu": False,
                # CLAHE для RGB (теней меньше влияют на YOLO)
                "clahe_enable": True,
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": 8,
            }]
        ),

        Node(
            package="grasp_inference_pkg",
            executable="model_forward",
            name="model_forward",
            output="screen",
            parameters=[{
                "color_topic": "/heightmap_node/heightmap/color",
                "height_topic": "/heightmap_node/heightmap/height",
                "mask_topic": "/heightmap_node/heightmap/mask",

                "model_path": model_path,
                "force_cpu": False,
                "grasp_depth_offset": 0.0,
                "sync_slop": 0.1,
                "hm_size": 224,
                "hm_resolution": 0.002,
                "plane_min": [-0.224, -0.224],
                "num_rotations": 16,
            }]
        )
    ])