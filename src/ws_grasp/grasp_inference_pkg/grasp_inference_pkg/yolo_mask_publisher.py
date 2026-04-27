from __future__ import annotations

from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def _is_bg_class(class_name: str) -> bool:
    name = (class_name or "").strip().lower()
    return name.startswith("bg")


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

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.seg_model_path = Path(str(self.get_parameter("seg_model_path").value)).expanduser()
        self.seg_imgsz = int(self.get_parameter("seg_imgsz").value)
        self.seg_conf = float(self.get_parameter("seg_conf").value)
        self.seg_iou = float(self.get_parameter("seg_iou").value)
        self.seg_device = "cpu" if bool(self.get_parameter("seg_force_cpu").value) else None

        if not self.seg_model_path.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.seg_model_path}")

        from ultralytics import YOLO

        self.model = YOLO(str(self.seg_model_path))
        names = getattr(self.model, "names", {})
        self.class_names = dict(names) if hasattr(names, "items") else dict(enumerate(names))

        self.pub_mask = self.create_publisher(Image, self.mask_topic, 10)
        self.create_subscription(Image, self.color_topic, self._on_color, qos_profile_sensor_data)

        self.get_logger().info(
            f"YOLO mask publisher ready: color={self.color_topic} mask={self.mask_topic} "
            f"weights={self.seg_model_path}"
        )

    def _on_color(self, msg: Image) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode color image: {exc}")
            return

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return

        prediction = self.model.predict(
            source=rgb[:, :, ::-1],
            imgsz=self.seg_imgsz,
            conf=self.seg_conf,
            iou=self.seg_iou,
            device=self.seg_device,
            verbose=False,
        )[0]

        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if prediction.masks is not None and prediction.masks.data is not None and len(prediction.masks.data) > 0:
            masks = prediction.masks.data
            valid_ix = list(range(len(masks)))
            if prediction.boxes is not None and len(prediction.boxes.cls) == len(masks) and self.class_names:
                class_ids = prediction.boxes.cls.cpu().numpy().astype(int)
                valid_ix = [
                    idx for idx in valid_ix
                    if not _is_bg_class(self.class_names.get(class_ids[idx], str(class_ids[idx])))
                ]
            if valid_ix:
                masks = masks[valid_ix]
                mask = (masks > 0.5).any(dim=0).to("cpu").numpy().astype(np.uint8) * 255
                if mask.shape != (height, width):
                    import cv2

                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        out.header = msg.header
        self.pub_mask.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloMaskPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
