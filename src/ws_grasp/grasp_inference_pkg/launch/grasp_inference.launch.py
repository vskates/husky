from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("grasp_inference_pkg")

    # 1) ЧЕКПОИНТ: имя файла я не могу знать заранее, поэтому тут дефолт.
    # Поменяйте "checkpoint_base.pt" на ваше реальное имя.
    model_path = os.path.join(pkg_share, "models", "model_9000.pth")

    seg_model_path = os.path.join(pkg_share, "models", "yolo_19.pt")

    # 2) TF-ФРЕЙМ: в сим-конфиге нет строки TF, поэтому дефолт.
    # Если у вас в TF база называется иначе — поменяйте.
    target_frame = "camera_link"

    # --- значения из вашего Isaac-конфига ---
    hm_size = 224
    hm_resolution = 0.002  # HEIGHTMAP_RESOLUTION
    plane_min = [-0.194, -0.064]
    plane_max = [0.254,0.384]
    grasp_depth_offset = 0.00  # GRASP_DEPTH (base mode)

    return LaunchDescription([
        Node(
            package="grasp_inference_pkg",
            executable="grasp_node",
            name="heightmap_node",
            output="screen",
            parameters=[{
                "pcd_topic": "/camera/camera/depth/color/points",
                "target_frame": target_frame,
                "pcd_mask_from_topic": "/camera/camera/color/image_raw",  # маска как в ~/debug/yolo_mask_on_image_raw → build_heightmaps

                "hm_size": hm_size,
                "hm_resolution": hm_resolution,
                "plane_min": plane_min,
                "plane_max": plane_max,
                "out_prefix": "heightmap",

                # YOLO
                "seg_model_path": seg_model_path,
                "seg_imgsz": 640,
                "seg_conf": 0.25,
                "seg_iou": 0.7,
                "seg_force_cpu": False,
                "seg_mask_persist_frames": 5,
                "accumulate_frames": 10,
                "min_coverage": 0.02,
            }]
        ),

        Node(
            package="grasp_inference_pkg",
            executable="model_forward",
            name="grasp_inference_node",
            output="screen",
            parameters=[{
                "color_topic": "/heightmap_node/heightmap/color",
                "height_topic": "/heightmap_node/heightmap/height",
                "mask_topic": "/heightmap_node/heightmap/mask",

                "model_path": model_path,
                "force_cpu": False,

                "hm_size": hm_size,
                "hm_resolution": hm_resolution,
                "plane_min": plane_min,
                "grasp_depth_offset": grasp_depth_offset,

                "sync_slop": 0.1,

                "q_blur_sigma": 3.0,
                "pose_ema_alpha": 0.4,

                "target_frame": "base",
                "transform_timeout": 1.0,
            }]
        )
    ])
