from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener

from .pointcloud_ros import (
    apply_transform,
    normalize_frame_id,
    read_xyz_points,
    transform_to_matrix,
)


@dataclass
class InferenceResult:
    grasps: np.ndarray
    scores: np.ndarray


@dataclass
class PreparedPointCloud:
    normalized_points: np.ndarray
    source_frame: str
    inference_frame: str
    center: np.ndarray
    source_from_inference: np.ndarray | None


@dataclass
class CandidateSelectionResult:
    success: bool
    selected_grasp: np.ndarray
    selected_score: float
    selected_index: int
    filtered_grasps: np.ndarray
    filtered_scores: np.ndarray
    selected_pregrasp: np.ndarray | None = None
    was_flipped: bool = False
    reason: str = ""


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (rotation[2, 1] - rotation[1, 2]) * s
        qy = (rotation[0, 2] - rotation[2, 0]) * s
        qz = (rotation[1, 0] - rotation[0, 1]) * s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    pose = Pose()
    quat = rotation_matrix_to_quaternion(matrix[:3, :3])
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    return pose


class DiffusionGraspNode(Node):
    def __init__(self) -> None:
        super().__init__("diffusion_grasp_node")

        self.declare_parameter("pointcloud_topic", "/segmented_object_pcd_node/points")
        self.declare_parameter("full_pointcloud_topic", "")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("inference_frame", "")
        self.declare_parameter("robot_base_frame", "base")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("transform_timeout", 0.5)
        self.declare_parameter("backend_mode", "auto")
        self.declare_parameter("graspgen_repo_path", "/home/weshi/graspgen")
        self.declare_parameter("gripper_config", "/home/weshi/graspgen/GraspGenModels/checkpoints/graspgen_astribot.yml")
        self.declare_parameter("conda_env_name", "isaaclab")
        self.declare_parameter("conda_executable", "")
        self.declare_parameter("subprocess_timeout_sec", 120.0)
        self.declare_parameter("force_cpu", False)
        self.declare_parameter("num_grasps", 500)
        self.declare_parameter("topk_num_grasps", 200)
        self.declare_parameter("min_grasps", 40)
        self.declare_parameter("max_tries", 6)
        self.declare_parameter("grasp_threshold", 0.0)
        self.declare_parameter("remove_outliers", True)
        self.declare_parameter("min_points", 20)
        self.declare_parameter("max_points", 2048)
        self.declare_parameter("min_inference_interval_sec", 0.0)
        self.declare_parameter("candidate_count_to_publish", 20)
        self.declare_parameter("candidate_filter_mode", "none")
        self.declare_parameter("candidate_filter_planning_frame", "")
        self.declare_parameter("curobo_robot_config_path", "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/curobo_configs/astribot/astribot.yml")
        self.declare_parameter("curobo_robot_info_path", "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/curobo_configs/astribot/robot_info.json")
        self.declare_parameter("curobo_ee_link_name", "astribot_gripper_right_base_link")
        self.declare_parameter("gripper_mesh_path", "/home/weshi/rl_grasp_pose/MetaIsaacGrasp/models/Gripper/Astribot/astribot_sphere.obj")
        self.declare_parameter(
            "arm_joint_names",
            [
                "astribot_arm_right_joint_1",
                "astribot_arm_right_joint_2",
                "astribot_arm_right_joint_3",
                "astribot_arm_right_joint_4",
                "astribot_arm_right_joint_5",
                "astribot_arm_right_joint_6",
                "astribot_arm_right_joint_7",
            ],
        )
        self.declare_parameter("filter_collision_threshold", 0.002)
        self.declare_parameter("filter_pregrasp_backoff", 0.1)
        self.declare_parameter("filter_pregrasp_halfspace_margin", 0.17)
        self.declare_parameter("filter_pregrasp_exit_eps", 1.0e-4)
        self.declare_parameter("filter_curobo_enabled", True)
        self.declare_parameter("filter_curobo_batch_size", 100)
        self.declare_parameter("filter_curobo_max_iters", 60)
        self.declare_parameter("filter_curobo_num_ik_seeds", 20)
        self.declare_parameter("filter_curobo_num_graph_seeds", 4)
        self.declare_parameter("filter_curobo_num_trajopt_seeds", 4)
        self.declare_parameter("filter_curobo_timeout", 1.0)
        self.declare_parameter("filter_curobo_world_voxel_size", 0.01)
        self.declare_parameter("filter_curobo_world_buffer", 0.005)
        self.declare_parameter("filter_curobo_world_crop_margin", 0.35)
        self.declare_parameter("filter_curobo_max_world_cuboids", 360)
        self.declare_parameter("filter_curobo_position_threshold", 0.01)
        self.declare_parameter("filter_curobo_rotation_threshold", 0.08)
        self.declare_parameter("gripper_translation_offset", [-0.096, 0.008, -0.12])
        self.declare_parameter("gripper_rpy_offset", [0.0, 0.0, 0.0])

        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
        self.full_pointcloud_topic = str(self.get_parameter("full_pointcloud_topic").value or "").strip()
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value or "").strip()
        self.inference_frame = normalize_frame_id(str(self.get_parameter("inference_frame").value))
        self.robot_base_frame = normalize_frame_id(str(self.get_parameter("robot_base_frame").value))
        self.camera_frame = normalize_frame_id(str(self.get_parameter("camera_frame").value))
        self.transform_timeout = float(self.get_parameter("transform_timeout").value)
        self.backend_mode = str(self.get_parameter("backend_mode").value).strip().lower()
        self.graspgen_repo_path = os.path.expanduser(str(self.get_parameter("graspgen_repo_path").value))
        self.gripper_config = os.path.expanduser(str(self.get_parameter("gripper_config").value))
        self.conda_env_name = str(self.get_parameter("conda_env_name").value)
        self.conda_executable = str(self.get_parameter("conda_executable").value or "").strip()
        self.subprocess_timeout_sec = float(self.get_parameter("subprocess_timeout_sec").value)
        self.force_cpu = bool(self.get_parameter("force_cpu").value)
        self.num_grasps = int(self.get_parameter("num_grasps").value)
        self.topk_num_grasps = int(self.get_parameter("topk_num_grasps").value)
        self.min_grasps = int(self.get_parameter("min_grasps").value)
        self.max_tries = int(self.get_parameter("max_tries").value)
        self.grasp_threshold = float(self.get_parameter("grasp_threshold").value)
        self.remove_outliers = bool(self.get_parameter("remove_outliers").value)
        self.min_points = int(self.get_parameter("min_points").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.min_inference_interval_sec = float(self.get_parameter("min_inference_interval_sec").value)
        self.candidate_count_to_publish = int(self.get_parameter("candidate_count_to_publish").value)
        self.candidate_filter_mode = str(self.get_parameter("candidate_filter_mode").value).strip().lower()
        self.candidate_filter_planning_frame = normalize_frame_id(
            str(self.get_parameter("candidate_filter_planning_frame").value)
        )
        self.curobo_robot_config_path = os.path.expanduser(str(self.get_parameter("curobo_robot_config_path").value))
        self.curobo_robot_info_path = os.path.expanduser(str(self.get_parameter("curobo_robot_info_path").value))
        self.curobo_ee_link_name = normalize_frame_id(str(self.get_parameter("curobo_ee_link_name").value))
        self.gripper_mesh_path = os.path.expanduser(str(self.get_parameter("gripper_mesh_path").value))
        self.arm_joint_names = [str(name) for name in self.get_parameter("arm_joint_names").value]
        self.filter_collision_threshold = float(self.get_parameter("filter_collision_threshold").value)
        self.filter_pregrasp_backoff = float(self.get_parameter("filter_pregrasp_backoff").value)
        self.filter_pregrasp_halfspace_margin = float(self.get_parameter("filter_pregrasp_halfspace_margin").value)
        self.filter_pregrasp_exit_eps = float(self.get_parameter("filter_pregrasp_exit_eps").value)
        self.filter_curobo_enabled = bool(self.get_parameter("filter_curobo_enabled").value)
        self.filter_curobo_batch_size = int(self.get_parameter("filter_curobo_batch_size").value)
        self.filter_curobo_max_iters = int(self.get_parameter("filter_curobo_max_iters").value)
        self.filter_curobo_num_ik_seeds = int(self.get_parameter("filter_curobo_num_ik_seeds").value)
        self.filter_curobo_num_graph_seeds = int(self.get_parameter("filter_curobo_num_graph_seeds").value)
        self.filter_curobo_num_trajopt_seeds = int(self.get_parameter("filter_curobo_num_trajopt_seeds").value)
        self.filter_curobo_timeout = float(self.get_parameter("filter_curobo_timeout").value)
        self.filter_curobo_world_voxel_size = float(self.get_parameter("filter_curobo_world_voxel_size").value)
        self.filter_curobo_world_buffer = float(self.get_parameter("filter_curobo_world_buffer").value)
        self.filter_curobo_world_crop_margin = float(self.get_parameter("filter_curobo_world_crop_margin").value)
        self.filter_curobo_max_world_cuboids = int(self.get_parameter("filter_curobo_max_world_cuboids").value)
        self.filter_curobo_position_threshold = float(self.get_parameter("filter_curobo_position_threshold").value)
        self.filter_curobo_rotation_threshold = float(self.get_parameter("filter_curobo_rotation_threshold").value)

        gripper_translation_offset = np.array(
            self.get_parameter("gripper_translation_offset").value,
            dtype=np.float32,
        )
        gripper_rpy_offset = np.array(self.get_parameter("gripper_rpy_offset").value, dtype=np.float32)
        self.gripper_offset = np.eye(4, dtype=np.float32)
        self.gripper_offset[:3, :3] = rpy_to_matrix(
            float(gripper_rpy_offset[0]),
            float(gripper_rpy_offset[1]),
            float(gripper_rpy_offset[2]),
        )
        self.gripper_offset[:3, 3] = gripper_translation_offset

        self.tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, "~/grasp_pose_base", 10)
        self.pub_grasp_pose_gripper = self.create_publisher(PoseStamped, "~/grasp_pose_gripper", 10)
        self.pub_grasp_candidates = self.create_publisher(PoseArray, "~/grasp_candidates", 10)
        self.pub_best_score = self.create_publisher(Float32, "~/best_score", 10)
        self.create_subscription(PointCloud2, self.pointcloud_topic, self._on_pointcloud, 10)
        if self.full_pointcloud_topic:
            self.create_subscription(PointCloud2, self.full_pointcloud_topic, self._on_full_pointcloud, 10)
        if self.joint_state_topic:
            self.create_subscription(JointState, self.joint_state_topic, self._on_joint_state, 10)

        self._last_inference_time = None
        self._backend_kind = ""
        self._python_sampler = None
        self._python_run_inference = None
        self._latest_full_cloud_msg: PointCloud2 | None = None
        self._latest_joint_state_msg: JointState | None = None
        self._setup_backend()

        self.get_logger().info(
            f"Diffusion grasp node ready: topic={self.pointcloud_topic} backend={self._backend_kind} "
            f"inference_frame={self.inference_frame or '[cloud]'} "
            f"base_frame={self.robot_base_frame} camera_frame={self.camera_frame} "
            f"candidate_filter={self.candidate_filter_mode}"
        )

    def _setup_backend(self) -> None:
        if self.backend_mode not in {"auto", "python", "subprocess"}:
            raise ValueError(f"unsupported backend_mode={self.backend_mode}")

        if self.backend_mode in {"auto", "python"}:
            try:
                self._setup_python_backend()
                self._backend_kind = "python"
                return
            except Exception as exc:
                if self.backend_mode == "python":
                    raise
                self.get_logger().warning(f"Python backend unavailable, falling back to subprocess: {exc}")

        self._backend_kind = "subprocess"

    def _setup_python_backend(self) -> None:
        if not os.path.isdir(self.graspgen_repo_path):
            raise FileNotFoundError(self.graspgen_repo_path)
        if self.graspgen_repo_path not in sys.path:
            sys.path.insert(0, self.graspgen_repo_path)

        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

        cfg = load_grasp_cfg(self.gripper_config)
        self._python_sampler = GraspGenSampler(cfg)
        self._python_run_inference = GraspGenSampler.run_inference

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp: Time):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=Duration(seconds=self.transform_timeout),
            )
        except Exception:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout),
            )

    def _on_full_pointcloud(self, msg: PointCloud2) -> None:
        self._latest_full_cloud_msg = msg

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state_msg = msg

    def _on_pointcloud(self, msg: PointCloud2) -> None:
        now = self.get_clock().now()
        if (
            self._last_inference_time is not None
            and self.min_inference_interval_sec > 0.0
            and (now - self._last_inference_time).nanoseconds < self.min_inference_interval_sec * 1e9
        ):
            return

        try:
            points = read_xyz_points(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to decode PointCloud2: {exc}")
            return
        valid_object_points = points[
            np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
        ].astype(np.float32, copy=False)

        cloud_frame = normalize_frame_id(msg.header.frame_id)
        stamp = Time.from_msg(msg.header.stamp)
        prepared = self._prepare_points_for_inference(points, cloud_frame, stamp)
        if prepared.normalized_points.shape[0] < self.min_points:
            self.get_logger().warning(
                f"Too few valid object points for inference: {prepared.normalized_points.shape[0]} < {self.min_points}"
            )
            return

        result = self._run_inference(prepared.normalized_points)
        if result.grasps.shape[0] == 0:
            self.get_logger().warning("Diffusion model returned no grasps")
            return

        grasps_source = self._restore_grasps_to_source_frame(result.grasps, prepared)
        output_frame = prepared.source_frame or prepared.inference_frame
        selection = self._select_grasp_candidate(
            grasps=grasps_source,
            scores=np.asarray(result.scores, dtype=np.float32),
            object_points=valid_object_points,
            output_frame=output_frame,
            stamp=stamp,
        )
        if not selection.success:
            self.get_logger().warning(f"Candidate filtering failed: {selection.reason}")
            return

        best_grasp = selection.selected_grasp
        best_score = float(selection.selected_score)
        gripper_pose = best_grasp @ self.gripper_offset

        self._publish_pose(self.pub_grasp_pose, best_grasp, output_frame, msg.header.stamp)
        self._publish_base_pose(self.pub_grasp_pose_base, best_grasp, output_frame, msg.header.stamp)
        self._publish_base_pose(self.pub_grasp_pose_gripper, gripper_pose, output_frame, msg.header.stamp)
        self._publish_candidates(selection.filtered_grasps, selection.filtered_scores, output_frame, msg.header.stamp)

        score_msg = Float32()
        score_msg.data = best_score
        self.pub_best_score.publish(score_msg)
        self._last_inference_time = now

        self.get_logger().info(
            f"Published filtered grasp score={best_score:.4f} "
            f"selected_index={selection.selected_index} "
            f"published_candidates={selection.filtered_grasps.shape[0]} "
            f"reason={selection.reason}"
        )

    def _prepare_points_for_inference(self, points: np.ndarray, source_frame: str, stamp: Time) -> PreparedPointCloud:
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
        points = points[valid]
        inference_frame = self._resolve_inference_frame(source_frame)
        source_from_inference = None

        if points.shape[0] == 0:
            return PreparedPointCloud(
                normalized_points=points.astype(np.float32),
                source_frame=source_frame or inference_frame,
                inference_frame=inference_frame,
                center=np.zeros((3,), dtype=np.float32),
                source_from_inference=source_from_inference,
            )

        frame_id = inference_frame
        if frame_id and source_frame and source_frame != frame_id:
            try:
                transform = self._lookup_transform(frame_id, source_frame, stamp)
                transform_matrix = transform_to_matrix(transform)
                points = apply_transform(points, transform_matrix)
                source_from_inference = np.linalg.inv(transform_matrix).astype(np.float32)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not transform point cloud {source_frame} -> {frame_id}: {exc}. "
                    "Running inference in source frame."
                )
                frame_id = source_frame
        else:
            frame_id = source_frame or frame_id

        center = points.mean(axis=0).astype(np.float32)
        normalized_points = (points - center[None, :]).astype(np.float32, copy=False)

        if self.max_points > 0 and normalized_points.shape[0] > self.max_points:
            choice = np.linspace(0, points.shape[0] - 1, self.max_points, dtype=np.int64)
            normalized_points = normalized_points[choice]

        return PreparedPointCloud(
            normalized_points=normalized_points.astype(np.float32, copy=False),
            source_frame=source_frame or frame_id,
            inference_frame=frame_id,
            center=center,
            source_from_inference=source_from_inference,
        )

    def _resolve_inference_frame(self, source_frame: str) -> str:
        return self.inference_frame or source_frame or self.robot_base_frame

    def _restore_grasps_to_source_frame(
        self,
        grasps: np.ndarray,
        prepared: PreparedPointCloud,
    ) -> np.ndarray:
        grasps = np.asarray(grasps, dtype=np.float32)
        if grasps.ndim != 3 or grasps.shape[-2:] != (4, 4):
            raise ValueError(f"expected grasp tensor of shape (N, 4, 4), got {grasps.shape}")
        if grasps.shape[0] == 0:
            return grasps.astype(np.float32, copy=False)

        restored = grasps.copy()
        restored[:, :3, 3] += prepared.center[None, :]
        if prepared.source_from_inference is not None:
            restored = self._left_multiply_poses(prepared.source_from_inference, restored)
        restored[:, 3, 3] = 1.0
        return restored.astype(np.float32, copy=False)

    @staticmethod
    def _left_multiply_poses(transform: np.ndarray, poses: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform, dtype=np.float32)
        poses = np.asarray(poses, dtype=np.float32)
        return np.einsum("ij,njk->nik", transform, poses, dtype=np.float32).astype(np.float32, copy=False)

    def _select_grasp_candidate(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        object_points: np.ndarray,
        output_frame: str,
        stamp: Time,
    ) -> CandidateSelectionResult:
        grasps = np.asarray(grasps, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        object_points = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)
        if grasps.shape[0] == 0 or scores.shape[0] == 0:
            return CandidateSelectionResult(
                success=False,
                selected_grasp=np.eye(4, dtype=np.float32),
                selected_score=float("-inf"),
                selected_index=-1,
                filtered_grasps=np.zeros((0, 4, 4), dtype=np.float32),
                filtered_scores=np.zeros((0,), dtype=np.float32),
                reason="no candidates to select from",
            )

        if self.candidate_filter_mode == "none":
            order = np.argsort(scores)[::-1]
            best_index = int(order[0])
            publish_order = order[: max(self.candidate_count_to_publish, 1)]
            return CandidateSelectionResult(
                success=True,
                selected_grasp=grasps[best_index].astype(np.float32, copy=False),
                selected_score=float(scores[best_index]),
                selected_index=best_index,
                filtered_grasps=grasps[publish_order].astype(np.float32, copy=False),
                filtered_scores=scores[publish_order].astype(np.float32, copy=False),
                reason="selected highest-scoring raw candidate",
            )

        if self.candidate_filter_mode != "subprocess":
            return CandidateSelectionResult(
                success=False,
                selected_grasp=np.eye(4, dtype=np.float32),
                selected_score=float("-inf"),
                selected_index=-1,
                filtered_grasps=np.zeros((0, 4, 4), dtype=np.float32),
                filtered_scores=np.zeros((0,), dtype=np.float32),
                reason=f"unsupported candidate_filter_mode={self.candidate_filter_mode}",
            )

        scene_points, scene_frame = self._get_scene_points()
        if scene_points.shape[0] == 0:
            scene_points = object_points
            scene_frame = output_frame

        planning_frame = self.candidate_filter_planning_frame or output_frame
        grasps_planning = grasps
        object_points_planning = object_points
        scene_points_planning = scene_points
        output_from_planning = None
        effective_frame = output_frame

        if planning_frame and output_frame and planning_frame != output_frame:
            try:
                transform = self._lookup_transform(planning_frame, output_frame, stamp)
                planning_from_output = transform_to_matrix(transform)
                grasps_planning = self._left_multiply_poses(planning_from_output, grasps)
                object_points_planning = apply_transform(object_points, planning_from_output)
                if scene_frame == output_frame:
                    scene_points_planning = apply_transform(scene_points, planning_from_output)
                else:
                    scene_points_planning = self._transform_points_between_frames(
                        scene_points,
                        scene_frame,
                        planning_frame,
                        stamp,
                    )
                output_from_planning = np.linalg.inv(planning_from_output).astype(np.float32)
                effective_frame = planning_frame
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not transform candidate filter inputs {output_frame} -> {planning_frame}: {exc}. "
                    "Falling back to the input cloud frame."
                )
                planning_frame = output_frame
        elif scene_frame and output_frame and scene_frame != output_frame:
            try:
                scene_points_planning = self._transform_points_between_frames(
                    scene_points,
                    scene_frame,
                    output_frame,
                    stamp,
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not transform full scene cloud {scene_frame} -> {output_frame}: {exc}. "
                    "Using segmented object cloud as the planning scene."
                )
                scene_points_planning = object_points

        current_ee_pose = self._lookup_current_ee_pose(planning_frame or output_frame, stamp)
        start_joint_positions = self._extract_start_joint_positions()

        try:
            selection = self._run_candidate_filter_subprocess(
                grasps=grasps_planning,
                scores=scores,
                object_points=object_points_planning,
                scene_points=scene_points_planning,
                start_joint_positions=start_joint_positions,
                current_ee_pose=current_ee_pose,
            )
        except Exception as exc:
            self.get_logger().warning(f"Candidate filter subprocess failed, falling back to raw ranking: {exc}")
            order = np.argsort(scores)[::-1]
            best_index = int(order[0])
            publish_order = order[: max(self.candidate_count_to_publish, 1)]
            return CandidateSelectionResult(
                success=True,
                selected_grasp=grasps[best_index].astype(np.float32, copy=False),
                selected_score=float(scores[best_index]),
                selected_index=best_index,
                filtered_grasps=grasps[publish_order].astype(np.float32, copy=False),
                filtered_scores=scores[publish_order].astype(np.float32, copy=False),
                reason=f"raw fallback after filter error: {exc}",
            )

        if not selection.success:
            return selection

        if output_from_planning is not None and effective_frame != output_frame:
            selection.selected_grasp = (output_from_planning @ selection.selected_grasp).astype(np.float32, copy=False)
            if selection.selected_pregrasp is not None:
                selection.selected_pregrasp = (
                    output_from_planning @ selection.selected_pregrasp
                ).astype(np.float32, copy=False)
            if selection.filtered_grasps.shape[0] > 0:
                selection.filtered_grasps = self._left_multiply_poses(
                    output_from_planning,
                    selection.filtered_grasps,
                )
        return selection

    def _get_scene_points(self) -> tuple[np.ndarray, str]:
        if self._latest_full_cloud_msg is None:
            return np.zeros((0, 3), dtype=np.float32), ""
        try:
            points = read_xyz_points(self._latest_full_cloud_msg)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32), ""
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
        return (
            points[valid].astype(np.float32, copy=False),
            normalize_frame_id(self._latest_full_cloud_msg.header.frame_id),
        )

    def _transform_points_between_frames(
        self,
        points: np.ndarray,
        source_frame: str,
        target_frame: str,
        stamp: Time,
    ) -> np.ndarray:
        if points.shape[0] == 0 or not source_frame or not target_frame or source_frame == target_frame:
            return points.astype(np.float32, copy=False)
        transform = self._lookup_transform(target_frame, source_frame, stamp)
        return apply_transform(points.astype(np.float32, copy=False), transform_to_matrix(transform))

    def _lookup_current_ee_pose(self, target_frame: str, stamp: Time) -> np.ndarray | None:
        if not target_frame or not self.curobo_ee_link_name:
            return None
        try:
            transform = self._lookup_transform(target_frame, self.curobo_ee_link_name, stamp)
        except Exception:
            return None
        return transform_to_matrix(transform)

    def _extract_start_joint_positions(self) -> list[float] | None:
        if self._latest_joint_state_msg is None or not self.arm_joint_names:
            return None
        name_to_position = {
            str(name): float(position)
            for name, position in zip(
                self._latest_joint_state_msg.name,
                self._latest_joint_state_msg.position,
                strict=False,
            )
        }
        positions: list[float] = []
        for joint_name in self.arm_joint_names:
            if joint_name not in name_to_position:
                return None
            positions.append(name_to_position[joint_name])
        return positions

    def _run_candidate_filter_subprocess(
        self,
        grasps: np.ndarray,
        scores: np.ndarray,
        object_points: np.ndarray,
        scene_points: np.ndarray,
        start_joint_positions: list[float] | None,
        current_ee_pose: np.ndarray | None,
    ) -> CandidateSelectionResult:
        helper_script = Path(__file__).with_name("curobo_candidate_filter_headless.py")
        if not helper_script.is_file():
            raise FileNotFoundError(f"candidate filter helper script not found: {helper_script}")

        with tempfile.TemporaryDirectory(prefix="grasp_filter_ros_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            grasps_path = tmp_path / "grasps.npy"
            scores_path = tmp_path / "scores.npy"
            object_points_path = tmp_path / "object_pc.npy"
            scene_points_path = tmp_path / "scene_pc.npy"
            config_path = tmp_path / "filter_config.json"
            output_path = tmp_path / "filter_output.npz"

            np.save(grasps_path, np.asarray(grasps, dtype=np.float32))
            np.save(scores_path, np.asarray(scores, dtype=np.float32))
            np.save(object_points_path, np.asarray(object_points, dtype=np.float32))
            np.save(scene_points_path, np.asarray(scene_points, dtype=np.float32))
            config_path.write_text(
                json.dumps(
                    {
                        "robot_config_path": self.curobo_robot_config_path,
                        "robot_info_path": self.curobo_robot_info_path,
                        "gripper_mesh_path": self.gripper_mesh_path,
                        "collision_threshold": self.filter_collision_threshold,
                        "pregrasp_backoff": self.filter_pregrasp_backoff,
                        "pregrasp_halfspace_margin": self.filter_pregrasp_halfspace_margin,
                        "pregrasp_exit_eps": self.filter_pregrasp_exit_eps,
                        "curobo_enabled": self.filter_curobo_enabled,
                        "curobo_batch_size": self.filter_curobo_batch_size,
                        "curobo_max_iters": self.filter_curobo_max_iters,
                        "curobo_num_ik_seeds": self.filter_curobo_num_ik_seeds,
                        "curobo_num_graph_seeds": self.filter_curobo_num_graph_seeds,
                        "curobo_num_trajopt_seeds": self.filter_curobo_num_trajopt_seeds,
                        "curobo_timeout": self.filter_curobo_timeout,
                        "curobo_world_voxel_size": self.filter_curobo_world_voxel_size,
                        "curobo_world_buffer": self.filter_curobo_world_buffer,
                        "curobo_world_crop_margin": self.filter_curobo_world_crop_margin,
                        "curobo_max_world_cuboids": self.filter_curobo_max_world_cuboids,
                        "curobo_position_threshold": self.filter_curobo_position_threshold,
                        "curobo_rotation_threshold": self.filter_curobo_rotation_threshold,
                        "start_joint_positions": start_joint_positions,
                        "current_ee_pose": None if current_ee_pose is None else np.asarray(current_ee_pose, dtype=np.float32).tolist(),
                        "force_cpu": self.force_cpu,
                    }
                ),
                encoding="utf-8",
            )

            cmd = [
                self.conda_executable or "conda",
                "run",
                "-n",
                self.conda_env_name,
                "python",
                str(helper_script),
                "--config",
                str(config_path),
                "--grasps",
                str(grasps_path),
                "--scores",
                str(scores_path),
                "--object-point-cloud",
                str(object_points_path),
                "--scene-point-cloud",
                str(scene_points_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_sec,
                env=self._subprocess_env(),
            )
            if completed.stdout:
                self.get_logger().info(completed.stdout.strip())
            archive = np.load(output_path, allow_pickle=True)
            success = bool(np.asarray(archive["success"]).item())
            reason = str(np.asarray(archive["reason"]).item())
            return CandidateSelectionResult(
                success=success,
                selected_grasp=np.asarray(archive["selected_grasp"], dtype=np.float32),
                selected_score=float(np.asarray(archive["selected_score"]).item()),
                selected_index=int(np.asarray(archive["selected_index"]).item()),
                filtered_grasps=np.asarray(archive["filtered_grasps"], dtype=np.float32),
                filtered_scores=np.asarray(archive["filtered_scores"], dtype=np.float32),
                selected_pregrasp=np.asarray(archive["selected_pregrasp"], dtype=np.float32),
                was_flipped=bool(np.asarray(archive["selected_was_flipped"]).item()),
                reason=reason,
            )

    def _run_inference(self, points: np.ndarray) -> InferenceResult:
        if self._backend_kind == "python":
            grasps, scores = self._python_run_inference(
                points,
                grasp_sampler=self._python_sampler,
                grasp_threshold=self.grasp_threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.topk_num_grasps,
                min_grasps=self.min_grasps,
                max_tries=self.max_tries,
                remove_outliers=self.remove_outliers,
            )
            if hasattr(grasps, "detach"):
                grasps = grasps.detach().cpu().numpy()
            if hasattr(scores, "detach"):
                scores = scores.detach().cpu().numpy()
            return InferenceResult(np.asarray(grasps, dtype=np.float32), np.asarray(scores, dtype=np.float32))

        with tempfile.TemporaryDirectory(prefix="graspgen_ros_") as tmp_dir:
            input_path = os.path.join(tmp_dir, "object_pc.npy")
            output_path = os.path.join(tmp_dir, "grasp_output.npz")
            np.save(input_path, points.astype(np.float32, copy=False))
            cmd = self._build_subprocess_command(input_path, output_path)
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout_sec,
                env=self._subprocess_env(),
            )
            if completed.stdout:
                self.get_logger().info(completed.stdout.strip())
            archive = np.load(output_path)
            return InferenceResult(
                np.asarray(archive["grasps"], dtype=np.float32),
                np.asarray(archive["scores"], dtype=np.float32),
            )

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{self.graspgen_repo_path}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = self.graspgen_repo_path
        if self.force_cpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def _build_subprocess_command(self, input_path: str, output_path: str) -> list[str]:
        script_path = os.path.join(self.graspgen_repo_path, "scripts", "inference_object_pc_headless.py")
        conda_executable = self.conda_executable or "conda"
        cmd = [
            conda_executable,
            "run",
            "-n",
            self.conda_env_name,
            "python",
            script_path,
            "--point-cloud",
            input_path,
            "--gripper-config",
            self.gripper_config,
            "--output",
            output_path,
            "--num-grasps",
            str(self.num_grasps),
            "--topk-num-grasps",
            str(self.topk_num_grasps),
            "--min-grasps",
            str(self.min_grasps),
            "--max-tries",
            str(self.max_tries),
            "--grasp-threshold",
            str(self.grasp_threshold),
        ]
        if not self.remove_outliers:
            cmd.append("--keep-outliers")
        return cmd

    def _publish_pose(self, publisher, pose_matrix: np.ndarray, frame_id: str, stamp) -> None:
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.pose = matrix_to_pose(pose_matrix)
        publisher.publish(msg)

    def _publish_base_pose(self, publisher, pose_matrix: np.ndarray, pose_frame: str, stamp) -> None:
        target_frame = self.robot_base_frame or pose_frame
        if not target_frame:
            return
        if target_frame == pose_frame:
            self._publish_pose(publisher, pose_matrix, pose_frame, stamp)
            return

        try:
            pose_base = self._transform_pose_matrix(
                pose_matrix,
                source_frame=pose_frame,
                target_frame=target_frame,
                stamp=Time.from_msg(stamp),
            )
            self._publish_pose(publisher, pose_base, target_frame, stamp)
        except Exception as exc:
            self.get_logger().warning(
                f"Could not publish base-frame grasp pose ({pose_frame} -> {target_frame}): {exc}"
            )

    def _transform_pose_matrix(
        self,
        pose_matrix: np.ndarray,
        source_frame: str,
        target_frame: str,
        stamp: Time,
    ) -> np.ndarray:
        if not source_frame or not target_frame or source_frame == target_frame:
            return pose_matrix.astype(np.float32, copy=False)

        transform = self._lookup_transform(target_frame, source_frame, stamp)
        transform_matrix = transform_to_matrix(transform)
        return (transform_matrix @ pose_matrix).astype(np.float32, copy=False)

    def _publish_candidates(self, grasps: np.ndarray, scores: np.ndarray, frame_id: str, stamp) -> None:
        grasps = np.asarray(grasps, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if grasps.shape[0] == 0:
            return

        order = np.argsort(scores)[::-1][: self.candidate_count_to_publish]
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = stamp
        pose_array.poses = [matrix_to_pose(grasps[int(idx)]) for idx in order]
        self.pub_grasp_candidates.publish(pose_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiffusionGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
