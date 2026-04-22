from __future__ import annotations

import numpy as np
import cv2

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import tf2_ros

from .pointcloud_ros import (
    R_OPTICAL_TO_LINK,
    T_OPTICAL_TO_LINK,
    normalize_frame_id,
    transform_to_matrix,
    xyz_to_pointcloud2,
)
from .projection import depth_to_xyz


class SegmentedObjectPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("segmented_object_pcd_node")
        self.bridge = CvBridge()

        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")
        self.declare_parameter("mask_topic", "")
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("transform_timeout", 0.2)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.0)
        self.declare_parameter("min_mask_value", 1)
        self.declare_parameter("fallback_optical_to_link", True)
        self.declare_parameter("publish_full_cloud", False)
        self.declare_parameter("max_publish_points", 25000)
        self.declare_parameter("output_topic", "~/points")
        self.declare_parameter("output_full_topic", "~/points_full")

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value or "").strip()
        self.target_frame = normalize_frame_id(str(self.get_parameter("target_frame").value))
        self.transform_timeout = float(self.get_parameter("transform_timeout").value)
        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.min_mask_value = int(self.get_parameter("min_mask_value").value)
        self.fallback_optical_to_link = bool(self.get_parameter("fallback_optical_to_link").value)
        self.publish_full_cloud = bool(self.get_parameter("publish_full_cloud").value)
        self.max_publish_points = int(self.get_parameter("max_publish_points").value)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_points = self.create_publisher(PointCloud2, str(self.get_parameter("output_topic").value), 10)
        self.pub_points_full = self.create_publisher(PointCloud2, str(self.get_parameter("output_full_topic").value), 10)

        self._sub_depth = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self._sub_info = message_filters.Subscriber(
            self, CameraInfo, self.camera_info_topic, qos_profile=qos_profile_sensor_data
        )

        if self.mask_topic:
            self._sub_mask = message_filters.Subscriber(
                self, Image, self.mask_topic, qos_profile=qos_profile_sensor_data
            )
            self._ts = message_filters.ApproximateTimeSynchronizer(
                [self._sub_depth, self._sub_info, self._sub_mask],
                queue_size=10,
                slop=0.1,
            )
            self._ts.registerCallback(self._on_depth_info_mask)
        else:
            self._ts = message_filters.ApproximateTimeSynchronizer(
                [self._sub_depth, self._sub_info],
                queue_size=10,
                slop=0.1,
            )
            self._ts.registerCallback(self._on_depth_info)

        self.get_logger().info(
            f"Segmented object PCD node ready: depth={self.depth_topic} camera_info={self.camera_info_topic} "
            f"mask={self.mask_topic or '[none]'} target_frame={self.target_frame or '[source]'}"
        )

    def _decode_depth(self, depth_msg: Image) -> np.ndarray:
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) * self.depth_scale
        else:
            depth = np.asarray(depth, dtype=np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[(depth < self.min_depth_m) | (depth > self.max_depth_m)] = np.nan
        return depth

    def _decode_mask(self, mask_msg: Image, target_shape: tuple[int, int]) -> np.ndarray:
        mask = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="passthrough")
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != target_shape and mask.T.shape == target_shape:
            mask = mask.T
        elif mask.shape != target_shape:
            mask = cv2.resize(mask.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask >= self.min_mask_value).astype(np.uint8)

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp: Time):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=stamp,
                timeout=Duration(seconds=self.transform_timeout),
            )
        except Exception:
            return self.tf_buffer.lookup_transform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=Time(),
                timeout=Duration(seconds=self.transform_timeout),
            )

    def _process(self, depth_msg: Image, info_msg: CameraInfo, mask_msg: Image | None) -> None:
        depth = self._decode_depth(depth_msg)
        mask = None if mask_msg is None else self._decode_mask(mask_msg, depth.shape)

        camera_frame = normalize_frame_id(info_msg.header.frame_id or depth_msg.header.frame_id)
        out_frame = camera_frame
        optical_to_link = None

        if self.target_frame and self.target_frame != camera_frame:
            stamp = Time.from_msg(depth_msg.header.stamp)
            try:
                transform = self._lookup_transform(self.target_frame, camera_frame, stamp)
                transform_matrix = transform_to_matrix(transform)
            except Exception as exc:
                if (
                    self.fallback_optical_to_link
                    and camera_frame.endswith("_optical_frame")
                    and self.target_frame == "camera_link"
                ):
                    transform_matrix = np.eye(4, dtype=np.float32)
                    transform_matrix[:3, :3] = R_OPTICAL_TO_LINK
                    transform_matrix[:3, 3] = T_OPTICAL_TO_LINK
                else:
                    self.get_logger().warning(
                        f"TF lookup failed ({camera_frame} -> {self.target_frame}): {exc}. "
                        "Publishing in source frame."
                    )
                    transform_matrix = None

            if transform_matrix is not None:
                optical_to_link = (transform_matrix[:3, :3], transform_matrix[:3, 3])
                out_frame = self.target_frame

        xyz_map = depth_to_xyz(
            depth,
            np.array(info_msg.k, dtype=np.float64).reshape(3, 3),
            frame_optical_to_link=optical_to_link,
        )

        valid_mask = np.isfinite(xyz_map).all(axis=-1)
        if mask is not None:
            segmented_mask = valid_mask & (mask > 0)
        else:
            segmented_mask = valid_mask

        full_points = xyz_map[valid_mask]
        object_points = xyz_map[segmented_mask]

        if full_points.shape[0] == 0:
            self.get_logger().warning("No valid depth points after filtering; skipping publish")
            return

        full_points = self._clip_points(full_points)
        object_points = self._clip_points(object_points)

        self.pub_points.publish(xyz_to_pointcloud2(object_points, out_frame, depth_msg.header.stamp))
        if self.publish_full_cloud:
            self.pub_points_full.publish(xyz_to_pointcloud2(full_points, out_frame, depth_msg.header.stamp))

        self.get_logger().info(
            f"Published object cloud: {object_points.shape[0]} points "
            f"(full={full_points.shape[0]}, mask={'on' if mask is not None else 'off'})"
        )

    def _clip_points(self, points: np.ndarray) -> np.ndarray:
        if self.max_publish_points <= 0 or points.shape[0] <= self.max_publish_points:
            return points.astype(np.float32, copy=False)
        choice = np.linspace(0, points.shape[0] - 1, self.max_publish_points, dtype=np.int64)
        return points[choice].astype(np.float32, copy=False)

    def _on_depth_info(self, depth_msg: Image, info_msg: CameraInfo) -> None:
        self._process(depth_msg, info_msg, None)

    def _on_depth_info_mask(self, depth_msg: Image, info_msg: CameraInfo, mask_msg: Image) -> None:
        self._process(depth_msg, info_msg, mask_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegmentedObjectPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
