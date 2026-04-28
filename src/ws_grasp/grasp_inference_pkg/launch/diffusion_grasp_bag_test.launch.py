from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bag_path = LaunchConfiguration("bag_path")
    seg_model_path = LaunchConfiguration("seg_model_path")
    frame_index = LaunchConfiguration("frame_index")

    return LaunchDescription(
        [
            DeclareLaunchArgument("bag_path", default_value="/home/weshi/HuskyRLGrasp/data"),
            DeclareLaunchArgument(
                "seg_model_path",
                default_value="/home/weshi/HuskyRLGrasp/models/yolo.pt",
            ),
            DeclareLaunchArgument("frame_index", default_value="0"),
            Node(
                package="grasp_inference_pkg",
                executable="bag_frame_publisher",
                name="bag_frame_publisher",
                output="screen",
                parameters=[
                    {
                        "bag_path": bag_path,
                        "frame_index": frame_index,
                        "publish_period_sec": 0.5,
                        "depth_topic": "/camera/camera/depth/image_rect_raw",
                        "depth_camera_info_topic": "/camera/camera/depth/camera_info",
                        "color_topic": "/camera/camera/color/image_rect_raw",
                        "color_camera_info_topic": "/camera/camera/color/camera_info",
                    }
                ],
            ),
            Node(
                package="grasp_inference_pkg",
                executable="yolo_mask_publisher",
                name="yolo_mask_publisher",
                output="screen",
                parameters=[
                    {
                        "color_topic": "/camera/camera/color/image_rect_raw",
                        "mask_topic": "/object_mask",
                        "seg_model_path": seg_model_path,
                        "seg_imgsz": 640,
                        "seg_conf": 0.25,
                        "seg_iou": 0.7,
                        "seg_force_cpu": False,
                        "conda_env_name": "isaaclab",
                        "conda_executable": "/home/weshi/miniconda3/bin/conda",
                    }
                ],
            ),
            Node(
                package="grasp_inference_pkg",
                executable="segmented_object_pcd_node",
                name="segmented_object_pcd_node",
                output="screen",
                parameters=[
                    {
                        "depth_topic": "/camera/camera/depth/image_rect_raw",
                        "camera_info_topic": "/camera/camera/depth/camera_info",
                        "mask_topic": "/object_mask",
                        "target_frame": "",
                        "publish_full_cloud": True,
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
                        "full_pointcloud_topic": "/segmented_object_pcd_node/points_full",
                        "joint_state_topic": "",
                        "inference_frame": "",
                        "robot_base_frame": "",
                        "camera_frame": "",
                        "backend_mode": "subprocess",
                        "candidate_filter_mode": "subprocess",
                        "candidate_filter_planning_frame": "",
                        "curobo_robot_config_path": "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/curobo_configs/astribot/astribot.yml",
                        "curobo_robot_info_path": "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/curobo_configs/astribot/robot_info.json",
                        "gripper_mesh_path": "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/models/Gripper/Astribot/astribot_sphere.obj",
                        "graspgen_repo_path": "/home/weshi/graspgen",
                        "gripper_config": "/home/weshi/graspgen/GraspGenModels/checkpoints/graspgen_astribot.yml",
                        "conda_env_name": "isaaclab",
                        "conda_executable": "/home/weshi/miniconda3/bin/conda",
                        "num_grasps": 500,
                        "topk_num_grasps": 200,
                        "min_grasps": 40,
                        "max_tries": 6,
                        "grasp_threshold": 0.0,
                        "remove_outliers": True,
                        "max_points": 2048,
                        "candidate_count_to_publish": 1,
                        "min_inference_interval_sec": 8.0,
                    }
                ],
            ),
        ]
    )
