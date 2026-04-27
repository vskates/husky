from __future__ import annotations

import copy
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import CameraInfo, Image


class BagFramePublisher(Node):
    def __init__(self) -> None:
        super().__init__("bag_frame_publisher")

        self.declare_parameter("bag_path", "")
        self.declare_parameter("frame_index", 0)
        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("depth_topic", "/camera/camera/depth/image_rect_raw")
        self.declare_parameter("depth_camera_info_topic", "/camera/camera/depth/camera_info")
        self.declare_parameter("color_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("color_camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("publish_color_camera_info", False)

        self.bag_path = Path(str(self.get_parameter("bag_path").value)).expanduser()
        self.frame_index = int(self.get_parameter("frame_index").value)
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.depth_camera_info_topic = str(self.get_parameter("depth_camera_info_topic").value)
        self.color_topic = str(self.get_parameter("color_topic").value)
        self.color_camera_info_topic = str(self.get_parameter("color_camera_info_topic").value)
        self.publish_color_camera_info = bool(self.get_parameter("publish_color_camera_info").value)

        self.pub_depth = self.create_publisher(Image, self.depth_topic, 10)
        self.pub_depth_info = self.create_publisher(CameraInfo, self.depth_camera_info_topic, 10)
        self.pub_color = self.create_publisher(Image, self.color_topic, 10)
        self.pub_color_info = self.create_publisher(CameraInfo, self.color_camera_info_topic, 10)

        self.depth_msg, self.depth_info_msg, self.color_msg, self.color_info_msg = self._load_frame()
        self.timer = self.create_timer(self.publish_period_sec, self._publish_frame)

        self.get_logger().info(
            f"Bag frame publisher ready: bag={self.bag_path} frame_index={self.frame_index} "
            f"topics=({self.depth_topic}, {self.depth_camera_info_topic}, {self.color_topic})"
        )

    def _bag_uri(self) -> str:
        if self.bag_path.is_dir():
            return str(self.bag_path)
        if self.bag_path.suffix == ".db3":
            return str(self.bag_path.parent)
        raise FileNotFoundError(f"Unsupported bag path: {self.bag_path}")

    def _load_frame(self):
        reader = SequentialReader()
        reader.open(
            StorageOptions(uri=self._bag_uri(), storage_id="sqlite3"),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )

        topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        buffers = {
            self.depth_topic: [],
            self.depth_camera_info_topic: [],
            self.color_topic: [],
            self.color_camera_info_topic: [],
        }
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic not in buffers:
                continue
            msg_type = get_message(topic_types[topic])
            buffers[topic].append(deserialize_message(data, msg_type))

        lengths = {topic: len(items) for topic, items in buffers.items()}
        min_frames = min(lengths[self.depth_topic], lengths[self.depth_camera_info_topic], lengths[self.color_topic])
        if min_frames == 0:
            raise RuntimeError(f"Bag does not contain required topics: {lengths}")
        if self.frame_index < 0 or self.frame_index >= min_frames:
            raise IndexError(f"frame_index={self.frame_index} out of range, available 0..{min_frames - 1}")

        color_info = buffers[self.color_camera_info_topic][min(self.frame_index, max(lengths[self.color_camera_info_topic] - 1, 0))] if lengths[self.color_camera_info_topic] > 0 else None
        return (
            buffers[self.depth_topic][self.frame_index],
            buffers[self.depth_camera_info_topic][self.frame_index],
            buffers[self.color_topic][self.frame_index],
            color_info,
        )

    def _publish_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        depth = copy.deepcopy(self.depth_msg)
        depth_info = copy.deepcopy(self.depth_info_msg)
        color = copy.deepcopy(self.color_msg)
        depth.header.stamp = stamp
        depth_info.header.stamp = stamp
        color.header.stamp = stamp
        self.pub_depth.publish(depth)
        self.pub_depth_info.publish(depth_info)
        self.pub_color.publish(color)

        if self.publish_color_camera_info and self.color_info_msg is not None:
            color_info = copy.deepcopy(self.color_info_msg)
            color_info.header.stamp = stamp
            self.pub_color_info.publish(color_info)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BagFramePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
