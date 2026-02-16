from __future__ import annotations

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2, PointField, Image
from cv_bridge import CvBridge

import tf2_ros
import cv2

from ultralytics import YOLO

from .projection import HeightmapSpec, build_heightmaps


def _norm_frame(s: str) -> str:
    s = (s or "").strip()
    while s.startswith("/"):
        s = s[1:]
    return s


def unpack_rgb_pcl_float(rgb_float: np.ndarray) -> np.ndarray:
    rgb_u32 = rgb_float.view(np.uint32)
    r = (rgb_u32 >> 16) & 0xFF
    g = (rgb_u32 >> 8) & 0xFF
    b = rgb_u32 & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def transform_to_matrix(t) -> np.ndarray:
    qx = t.transform.rotation.x
    qy = t.transform.rotation.y
    qz = t.transform.rotation.z
    qw = t.transform.rotation.w
    tx = t.transform.translation.x
    ty = t.transform.translation.y
    tz = t.transform.translation.z

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


R_OPTICAL_TO_LINK = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
T_OPTICAL_TO_LINK = np.zeros((3,), dtype=np.float32)


class HeightmapNode(Node):
    def __init__(self) -> None:
        super().__init__("heightmap_node")
        self.bridge = CvBridge()

        # ---- params ----
        self.declare_parameter("pcd_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("target_frame", "camera_link")

        self.declare_parameter("hm_size", 224)
        self.declare_parameter("hm_resolution", 0.001)
        self.declare_parameter("plane_min", [-0.224, -0.224])
        self.declare_parameter("plane_max", [0.224, 0.224])

        self.declare_parameter("out_prefix", "heightmap")
        self.declare_parameter("fallback_optical_to_link", True)

        # ---- YOLOv8-seg params ----s
        self.declare_parameter("seg_model_path", "yolo_finetuned.pt")
        self.declare_parameter("seg_imgsz", 640)
        self.declare_parameter("seg_conf", 0.25)
        self.declare_parameter("seg_iou", 0.7)
        self.declare_parameter("seg_force_cpu", False)

        self.pcd_topic = self.get_parameter("pcd_topic").value
        self.target_frame = _norm_frame(self.get_parameter("target_frame").value)
        self.out_prefix = self.get_parameter("out_prefix").value
        self.fallback_opt2link = bool(self.get_parameter("fallback_optical_to_link").value)

        S = int(self.get_parameter("hm_size").value)
        res = float(self.get_parameter("hm_resolution").value)
        plane_min = np.array(self.get_parameter("plane_min").value, dtype=np.float32)
        plane_max = np.array(self.get_parameter("plane_max").value, dtype=np.float32)

        self.spec = HeightmapSpec(
            size=S,
            resolution=res,
            plane_min=plane_min,
            plane_max=plane_max,
            height_axis=0,      # X
            plane_axes=(1, 2),  # (Y,Z)
        )

        # ---- YOLO init ----
        seg_model_path = self.get_parameter("seg_model_path").value
        self.seg_imgsz = int(self.get_parameter("seg_imgsz").value)
        self.seg_conf = float(self.get_parameter("seg_conf").value)
        self.seg_iou = float(self.get_parameter("seg_iou").value)
        seg_force_cpu = bool(self.get_parameter("seg_force_cpu").value)
        self.seg_device = "cpu" if seg_force_cpu else ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.seg = YOLO(seg_model_path)
        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_error_printed = False

        # ---- pubs ----
        self.pub_hm_color = self.create_publisher(Image, f"~/{self.out_prefix}/color", 10)
        self.pub_hm_height = self.create_publisher(Image, f"~/{self.out_prefix}/height", 10)
        self.pub_hm_vis = self.create_publisher(Image, f"~/{self.out_prefix}/height_vis", 10)
        self.pub_hm_mask = self.create_publisher(Image, f"~/{self.out_prefix}/mask", 10)

        # ---- sub ----
        self.create_subscription(PointCloud2, self.pcd_topic, self._on_pcd, qos_profile_sensor_data)

        self.get_logger().info(f"PCD topic: {self.pcd_topic}")
        self.get_logger().info(f"target_frame: '{self.target_frame or '[use msg.frame_id]'}'")
        self.get_logger().info("Heightmap mapping: height_axis=X, plane_axes=(Y,Z)")
        self.get_logger().info(f"Seg model: {seg_model_path} | imgsz={self.seg_imgsz} conf={self.seg_conf} iou={self.seg_iou}")

    def _try_lookup_tf(self, target: str, source: str, stamp: Time):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=stamp,
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            return self.tf_buffer.lookup_transform(
                target_frame=target,
                source_frame=source,
                time=Time(),
                timeout=Duration(seconds=0.2),
            )

    def _segment_union_mask(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Return HxW uint8 mask: 0=background, 255=object (union of all instance masks)."""
        img_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        r = self.seg.predict(
            source=img_bgr,
            imgsz=self.seg_imgsz,
            conf=self.seg_conf,
            iou=self.seg_iou,
            device=self.seg_device,
            verbose=False,
        )[0]

        H, W = rgb_u8.shape[:2]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return np.zeros((H, W), dtype=np.uint8)

        m = r.masks.data  # (N,h,w) torch
        union = (m > 0.5).any(dim=0).to("cpu").numpy().astype(np.uint8) * 255
        if union.shape != (H, W):
            union = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST)
        return union

    def _on_pcd(self, msg: PointCloud2) -> None:
        if msg.height <= 1:
            self.get_logger().error("PointCloud2 is not organized (height <= 1).")
            return

        src_frame = _norm_frame(msg.header.frame_id)
        tgt_frame = self.target_frame

        field_map = {f.name: f for f in msg.fields}
        for name in ("x", "y", "z"):
            if name not in field_map or field_map[name].datatype != PointField.FLOAT32:
                self.get_logger().error(f"PointCloud2 field '{name}' must exist and be FLOAT32")
                return

        has_rgb = ("rgb" in field_map) and (field_map["rgb"].datatype == PointField.FLOAT32)

        H, W = msg.height, msg.width
        n = H * W

        names = ["x", "y", "z"] + (["rgb"] if has_rgb else [])
        offsets = [int(field_map["x"].offset), int(field_map["y"].offset), int(field_map["z"].offset)]
        if has_rgb:
            offsets.append(int(field_map["rgb"].offset))

        dtype = np.dtype(
            {
                "names": names,
                "formats": [np.float32] * len(names),
                "offsets": offsets,
                "itemsize": int(msg.point_step),
            }
        )
        arr = np.frombuffer(msg.data, dtype=dtype, count=n)

        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).reshape(H, W, 3).astype(np.float32)
        rgb = unpack_rgb_pcl_float(arr["rgb"]).reshape(H, W, 3) if has_rgb else np.zeros((H, W, 3), dtype=np.uint8)

        # --- transform to target_frame ---
        out_frame = src_frame
        if tgt_frame and tgt_frame != src_frame:
            stamp = Time.from_msg(msg.header.stamp)
            transformed = False

            try:
                tf = self._try_lookup_tf(tgt_frame, src_frame, stamp)
                T = transform_to_matrix(tf)
                R = T[:3, :3]
                t = T[:3, 3]
                transformed = True
            except Exception as e:
                if not self._tf_error_printed:
                    self._tf_error_printed = True
                    self.get_logger().error(
                        f"TF lookup failed ({src_frame} -> {tgt_frame}): {e}\n"
                        f"Will {'use fallback optical->link rotation' if self.fallback_opt2link else 'skip transform'}."
                    )
                if self.fallback_opt2link and src_frame.endswith("_optical_frame") and tgt_frame == "camera_link":
                    R = R_OPTICAL_TO_LINK
                    t = T_OPTICAL_TO_LINK
                    transformed = True

            if transformed:
                xyz_flat = xyz.reshape(-1, 3)
                finite = np.isfinite(xyz_flat).all(axis=1)
                pts = xyz_flat[finite]
                pts_t = (pts @ R.T) + t[None, :]
                xyz_flat_out = xyz_flat.copy()
                xyz_flat_out[finite] = pts_t
                xyz = xyz_flat_out.reshape(H, W, 3)
                out_frame = tgt_frame

        # ---- segmentation mask on image plane (H,W) ----
        mask_u8 = self._segment_union_mask(rgb)

        # ---- build heightmaps (color, height, mask_heightmap) ----
        color_hm, height_hm, mask_hm = build_heightmaps(
            rgb_u8=rgb, xyz=xyz, spec=self.spec, mask_u8=mask_u8
        )

        # ---- publish ----
        msg_color = self.bridge.cv2_to_imgmsg(color_hm, encoding="rgb8")
        msg_color.header.stamp = msg.header.stamp
        msg_color.header.frame_id = out_frame
        self.pub_hm_color.publish(msg_color)

        msg_height = self.bridge.cv2_to_imgmsg(height_hm.astype(np.float32), encoding="32FC1")
        msg_height.header.stamp = msg.header.stamp
        msg_height.header.frame_id = out_frame
        self.pub_hm_height.publish(msg_height)

        # height_vis
        hm = height_hm
        finite_h = np.isfinite(hm)
        vis = np.zeros_like(hm, dtype=np.uint8)
        if finite_h.any():
            vmin = float(np.nanmin(hm[finite_h]))
            vmax = float(np.nanmax(hm[finite_h]))
            if vmax > vmin:
                vis = (np.clip((hm - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)

        msg_vis = self.bridge.cv2_to_imgmsg(vis, encoding="mono8")
        msg_vis.header.stamp = msg.header.stamp
        msg_vis.header.frame_id = out_frame
        self.pub_hm_vis.publish(msg_vis)

        # mask_heightmap (mono8 for visualization)
        mask_vis = (mask_hm > 0).astype(np.uint8) * 255
        msg_mask = self.bridge.cv2_to_imgmsg(mask_vis, encoding="mono8")
        msg_mask.header.stamp = msg.header.stamp
        msg_mask.header.frame_id = out_frame
        self.pub_hm_mask.publish(msg_mask)


def main() -> None:
    rclpy.init()
    node = HeightmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
