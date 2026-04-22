from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="grasp_inference_pkg",
                executable="segmented_object_pcd_node",
                name="segmented_object_pcd_node",
                output="screen",
                parameters=[
                    {
                        "depth_topic": "/camera/camera/aligned_depth_to_color/image_rect_raw",
                        "camera_info_topic": "/camera/camera/aligned_depth_to_color/camera_info",
                        "mask_topic": "/object_mask",
                        "target_frame": "base",
                        "publish_full_cloud": False,
                    }
                ],
            ),
            Node(
                package="grasp_inference_pkg",
                executable="diffusion_grasp_node",
                name="diffusion_grasp_node",
                output="screen",
                parameters=[
                    {
                        "pointcloud_topic": "/segmented_object_pcd_node/points",
                        "robot_base_frame": "base",
                        "camera_frame": "camera_link",
                        "backend_mode": "auto",
                        "graspgen_repo_path": "/home/weshi/graspgen",
                        "gripper_config": "/home/weshi/graspgen/GraspGenModels/checkpoints/graspgen_astribot.yml",
                        "conda_env_name": "graspgen-infer",
                        "num_grasps": 500,
                        "topk_num_grasps": 200,
                        "min_grasps": 40,
                        "max_tries": 6,
                        "grasp_threshold": 0.0,
                        "remove_outliers": True,
                        "max_points": 2048,
                    }
                ],
            ),
        ]
    )
