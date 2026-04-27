from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class YoloMaskPublisher(Node):
    def __init__(self) -> None:
        super().__init__("yolo_mask_publisher")
        self.bridge = CvBridge()

        self.declare_parameter("color_topic", "/camera/camera/color/image_rect_raw")
        self.declare_parameter("mask_topic", "/object_mask")
        self.declare_parameter("seg_model_path", "")
        self.declare_parameter("seg_imgsz", 640)
        self.declare_parameter("seg_conf", 0.25)
        self.declare_parameter("seg_iou", 0.7)
        self.declare_parameter("seg_force_cpu", False)
        self.declare_parameter("conda_env_name", "isaaclab")
        self.declare_parameter("conda_executable", "conda")
        self.declare_parameter("subprocess_timeout_sec", 120.0)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.seg_model_path = Path(str(self.get_parameter("seg_model_path").value)).expanduser()
        self.seg_imgsz = int(self.get_parameter("seg_imgsz").value)
        self.seg_conf = float(self.get_parameter("seg_conf").value)
        self.seg_iou = float(self.get_parameter("seg_iou").value)
        self.seg_force_cpu = bool(self.get_parameter("seg_force_cpu").value)
        self.conda_env_name = str(self.get_parameter("conda_env_name").value)
        self.conda_executable = str(self.get_parameter("conda_executable").value)
        self.subprocess_timeout_sec = float(self.get_parameter("subprocess_timeout_sec").value)

        if not self.seg_model_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.seg_model_path}")

        self.helper_script = Path(__file__).with_name("yolo_mask_inference.py")
        if not self.helper_script.is_file():
            raise FileNotFoundError(f"YOLO helper script not found: {self.helper_script}")

        self.pub_mask = self.create_publisher(Image, self.mask_topic, 10)
        self.create_subscription(Image, self.color_topic, self._on_color, qos_profile_sensor_data)

        self.get_logger().info(
            f"YOLO mask publisher ready: color={self.color_topic} mask={self.mask_topic} "
            f"weights={self.seg_model_path} conda_env={self.conda_env_name}"
        )

    def _build_subprocess_command(self, input_path: Path, output_path: Path) -> list[str]:
        cmd = [
            self.conda_executable,
            "run",
            "-n",
            self.conda_env_name,
            "python",
            str(self.helper_script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--weights",
            str(self.seg_model_path),
            "--imgsz",
            str(self.seg_imgsz),
            "--conf",
            str(self.seg_conf),
            "--iou",
            str(self.seg_iou),
        ]
        if self.seg_force_cpu:
            cmd.extend(["--device", "cpu"])
        return cmd

    def _on_color(self, msg: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode color image: {exc}")
            return

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return

        with tempfile.TemporaryDirectory(prefix="yolo_mask_") as tmp_dir:
            input_path = Path(tmp_dir) / "input.png"
            output_path = Path(tmp_dir) / "mask.png"
            if not cv2.imwrite(str(input_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                self.get_logger().warning(f"Failed to write temporary image: {input_path}")
                return

            try:
                completed = subprocess.run(
                    self._build_subprocess_command(input_path, output_path),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout_sec,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else str(exc)
                self.get_logger().error(f"YOLO subprocess failed: {stderr}")
                return
            except subprocess.TimeoutExpired:
                self.get_logger().error("YOLO subprocess timed out")
                return

            if completed.stderr:
                self.get_logger().debug(completed.stderr.strip())

            mask = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                self.get_logger().error(f"Failed to read generated mask: {output_path}")
                return

        out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        out.header = msg.header
        self.pub_mask.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloMaskPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
