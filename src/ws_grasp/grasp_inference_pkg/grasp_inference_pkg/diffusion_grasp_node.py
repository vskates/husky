from __future__ import annotations

import os
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
from sensor_msgs.msg import PointCloud2
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
        self.declare_parameter("inference_frame", "")
        self.declare_parameter("robot_base_frame", "base")
        self.declare_parameter("camera_frame", "camera_link")
        self.declare_parameter("transform_timeout", 0.5)
        self.declare_parameter("backend_mode", "auto")
        self.declare_parameter("graspgen_repo_path", "/home/weshi/graspgen")
        self.declare_parameter("gripper_config", "/home/weshi/graspgen/GraspGenModels/checkpoints/graspgen_astribot.yml")
        self.declare_parameter("conda_env_name", "graspgen-infer")
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
        self.declare_parameter("gripper_translation_offset", [-0.096, 0.008, -0.12])
        self.declare_parameter("gripper_rpy_offset", [0.0, 0.0, 0.0])

        self.pointcloud_topic = str(self.get_parameter("pointcloud_topic").value)
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
        self.pub_grasp_pose_camera = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, "~/grasp_pose_base", 10)
        self.pub_grasp_pose_gripper = self.create_publisher(PoseStamped, "~/grasp_pose_gripper", 10)
        self.pub_grasp_candidates = self.create_publisher(PoseArray, "~/grasp_candidates", 10)
        self.pub_best_score = self.create_publisher(Float32, "~/best_score", 10)
        self.create_subscription(PointCloud2, self.pointcloud_topic, self._on_pointcloud, 10)

        self._last_inference_time = None
        self._backend_kind = ""
        self._python_sampler = None
        self._python_run_inference = None
        self._setup_backend()

        self.get_logger().info(
            f"Diffusion grasp node ready: topic={self.pointcloud_topic} backend={self._backend_kind} "
            f"inference_frame={self.inference_frame or '[cloud]'} "
            f"base_frame={self.robot_base_frame} camera_frame={self.camera_frame}"
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

        cloud_frame = normalize_frame_id(msg.header.frame_id)
        stamp = Time.from_msg(msg.header.stamp)
        points, inference_frame = self._prepare_points_for_inference(points, cloud_frame, stamp)
        if points.shape[0] < self.min_points:
            self.get_logger().warning(
                f"Too few valid object points for inference: {points.shape[0]} < {self.min_points}"
            )
            return

        result = self._run_inference(points)
        if result.grasps.shape[0] == 0:
            self.get_logger().warning("Diffusion model returned no grasps")
            return

        best_index = int(np.argmax(result.scores))
        best_grasp = result.grasps[best_index]
        best_score = float(result.scores[best_index])
        gripper_pose = best_grasp @ self.gripper_offset

        self._publish_camera_pose(best_grasp, inference_frame, msg.header.stamp)
        self._publish_base_pose(self.pub_grasp_pose_base, best_grasp, inference_frame, msg.header.stamp)
        self._publish_base_pose(self.pub_grasp_pose_gripper, gripper_pose, inference_frame, msg.header.stamp)
        self._publish_candidates(result, inference_frame, msg.header.stamp)

        score_msg = Float32()
        score_msg.data = best_score
        self.pub_best_score.publish(score_msg)
        self._last_inference_time = now

        self.get_logger().info(
            f"Published best grasp score={best_score:.4f} total_candidates={result.grasps.shape[0]}"
        )

    def _prepare_points_for_inference(self, points: np.ndarray, source_frame: str, stamp: Time) -> tuple[np.ndarray, str]:
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
        points = points[valid]
        if points.shape[0] == 0:
            return points.astype(np.float32), self._resolve_inference_frame(source_frame)

        frame_id = self._resolve_inference_frame(source_frame)
        if frame_id and source_frame and source_frame != frame_id:
            try:
                transform = self._lookup_transform(frame_id, source_frame, stamp)
                transform_matrix = transform_to_matrix(transform)
                points = apply_transform(points, transform_matrix)
            except Exception as exc:
                self.get_logger().warning(
                    f"Could not transform point cloud {source_frame} -> {frame_id}: {exc}. "
                    "Running inference in source frame."
                )
                frame_id = source_frame
        else:
            frame_id = source_frame or frame_id

        if self.max_points > 0 and points.shape[0] > self.max_points:
            choice = np.linspace(0, points.shape[0] - 1, self.max_points, dtype=np.int64)
            points = points[choice]

        return points.astype(np.float32, copy=False), frame_id

    def _resolve_inference_frame(self, source_frame: str) -> str:
        return self.inference_frame or source_frame or self.robot_base_frame

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

    def _publish_camera_pose(self, pose_base: np.ndarray, pose_frame: str, stamp) -> None:
        if not self.camera_frame or self.camera_frame == pose_frame:
            self._publish_pose(self.pub_grasp_pose_camera, pose_base, pose_frame, stamp)
            return

        try:
            pose_camera = self._transform_pose_matrix(
                pose_base,
                source_frame=pose_frame,
                target_frame=self.camera_frame,
                stamp=Time.from_msg(stamp),
            )
            self._publish_pose(self.pub_grasp_pose_camera, pose_camera, self.camera_frame, stamp)
        except Exception as exc:
            self.get_logger().warning(
                f"Could not publish camera-frame grasp pose ({pose_frame} -> {self.camera_frame}): {exc}"
            )

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

    def _publish_candidates(self, result: InferenceResult, frame_id: str, stamp) -> None:
        if result.grasps.shape[0] == 0:
            return

        order = np.argsort(result.scores)[::-1][: self.candidate_count_to_publish]
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = stamp
        pose_array.poses = [matrix_to_pose(result.grasps[int(idx)]) for idx in order]
        self.pub_grasp_candidates.publish(pose_array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiffusionGraspNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
