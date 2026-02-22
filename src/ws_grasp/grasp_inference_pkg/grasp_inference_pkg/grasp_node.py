from __future__ import annotations

import time
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


def _is_bg_class(name: str) -> bool:
    """Классы с меткой bg* считаются фоном: не входят в маску, не показываются."""
    return (name or "").strip().lower().startswith("bg")


def _mask_has_pixel_above_line(mask: np.ndarray, H: int, W: int, cutoff_row: int) -> bool:
    """True, если у маски есть хотя бы один пиксель в области row < cutoff_row (выше линии)."""
    if cutoff_row <= 0:
        return True
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    top = min(cutoff_row, mask.shape[0])
    return bool(np.any(mask[:top, :] > 0))


def unpack_rgb_pcl_float(rgb_float: np.ndarray) -> np.ndarray:
    rgb_u32 = rgb_float.view(np.uint32)
    r = (rgb_u32 >> 16) & 0xFF
    g = (rgb_u32 >> 8) & 0xFF
    b = rgb_u32 & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def apply_clahe_rgb(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """CLAHE по каналу L в LAB; вход/выход RGB (H,W,3) uint8. Улучшает контраст в тенях."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a, b_ch])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB)


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

# Палитра BGR для масок классов (разные оттенки для подписей)
_CLASS_COLORS_BGR = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (0, 128, 255),
]


class HeightmapNode(Node):
    def __init__(self) -> None:
        super().__init__("heightmap_node")
        self.bridge = CvBridge()

        # ---- params ----
        self.declare_parameter("pcd_topic", "/camera/camera/depth/color/points")
        self.declare_parameter("target_frame", "camera_link")

        self.declare_parameter("hm_size", 224)
        self.declare_parameter("hm_resolution", 0.001)
        self.declare_parameter("plane_min", [0.438, 0.888])
        self.declare_parameter("plane_max", [0.662, 1.112])

        self.declare_parameter("out_prefix", "heightmap")
        self.declare_parameter("fallback_optical_to_link", True)

        # ---- YOLOv8-seg params ----
        self.declare_parameter("seg_model_path", "yolo_finetuned.pt")
        self.declare_parameter("seg_imgsz", 640)
        self.declare_parameter("seg_conf", 0.25)
        self.declare_parameter("seg_iou", 0.7)
        self.declare_parameter("seg_force_cpu", False)
        self.declare_parameter("seg_mask_persist_frames", 5)

        # ---- frame accumulation params ----
        self.declare_parameter("accumulate_frames", 10)
        self.declare_parameter("min_coverage", 0.02)

        # ---- линия отсечения гриппера: экземпляр оставляем только если у него есть пиксель выше линии ----
        # mask_above_line_fraction: доля высоты кадра (0..1). Верхние fraction остаются, низ отсекается. 1.0 = отключено
        self.declare_parameter("mask_above_line_fraction", 1.0)

        # ---- CLAHE для RGB (уменьшение влияния теней на YOLO) ----
        self.declare_parameter("clahe_enable", True)
        self.declare_parameter("clahe_clip_limit", 2.0)
        self.declare_parameter("clahe_tile_grid_size", 8)

        # ---- debug: publish YOLO mask on image plane ----
        self.declare_parameter("debug_publish_yolo_mask", True)
        self.declare_parameter(
            "debug_image_raw_topic", ""
        )  # e.g. "/camera/camera/color/image_raw" — маска накладывается и на image_raw

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
        self.seg_mask_persist = int(self.get_parameter("seg_mask_persist_frames").value)
        self.mask_above_line_fraction = float(self.get_parameter("mask_above_line_fraction").value)

        self._clahe_enable = bool(self.get_parameter("clahe_enable").value)
        self._clahe_clip_limit = float(self.get_parameter("clahe_clip_limit").value)
        self._clahe_tile_grid_size = int(self.get_parameter("clahe_tile_grid_size").value)

        # Классы модели: id -> имя (для дебага)
        try:
            names = getattr(self.seg.model, "names", None) or getattr(self.seg.predictor, "names", None)
            if names is not None:
                self._yolo_class_names = dict(names) if hasattr(names, "items") else dict(enumerate(names))
            else:
                self._yolo_class_names = {}
        except Exception:
            self._yolo_class_names = {}

        # ---- frame accumulation ----
        self.accumulate_n = int(self.get_parameter("accumulate_frames").value)
        self.min_coverage = float(self.get_parameter("min_coverage").value)

        S = self.spec.size
        self._acc_color = np.zeros((S, S, 3), dtype=np.uint8)
        self._acc_height = np.zeros((S, S), dtype=np.float32)
        self._acc_mask = np.zeros((S, S), dtype=np.uint8)
        self._acc_count: int = 0

        # ---- mask temporal smoothing только для PCD (для image_raw — без кэша, в реальном времени) ----
        self._last_valid_mask: np.ndarray | None = None
        self._mask_miss_count: int = 0
        self._last_yolo_classes_log: float = 0.0

        # ---- TF ----
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_error_printed = False

        # ---- pubs ----
        self.pub_hm_color = self.create_publisher(Image, f"~/{self.out_prefix}/color", 10)
        self.pub_hm_height = self.create_publisher(Image, f"~/{self.out_prefix}/height", 10)
        self.pub_hm_vis = self.create_publisher(Image, f"~/{self.out_prefix}/height_vis", 10)
        self.pub_hm_mask = self.create_publisher(Image, f"~/{self.out_prefix}/mask", 10)

        self._debug_yolo = bool(self.get_parameter("debug_publish_yolo_mask").value)
        self.pub_debug_yolo_rgb = self.create_publisher(Image, "~/debug/yolo_mask_on_rgb", 10)

        self._debug_image_raw_topic = (self.get_parameter("debug_image_raw_topic").value or "").strip()
        if self._debug_yolo and self._debug_image_raw_topic:
            self.pub_debug_yolo_on_image_raw = self.create_publisher(
                Image, "~/debug/yolo_mask_on_image_raw", 10
            )
            self.create_subscription(
                Image,
                self._debug_image_raw_topic,
                self._on_image_raw,
                10,
            )

        # ---- sub ----
        self.create_subscription(PointCloud2, self.pcd_topic, self._on_pcd, qos_profile_sensor_data)

        self.get_logger().info(f"PCD topic: {self.pcd_topic}")
        self.get_logger().info(f"target_frame: '{self.target_frame or '[use msg.frame_id]'}'")
        self.get_logger().info("Heightmap mapping: height_axis=X, plane_axes=(Y,Z)")
        self.get_logger().info(
            f"Seg model: {seg_model_path} | imgsz={self.seg_imgsz} "
            f"conf={self.seg_conf} iou={self.seg_iou} "
            f"mask_persist={self.seg_mask_persist} frames"
        )
        if self._yolo_class_names:
            self.get_logger().info(
                f"YOLO классы ({len(self._yolo_class_names)}): {self._yolo_class_names}"
            )
        if 0.0 < self.mask_above_line_fraction < 1.0:
            self.get_logger().info(
                f"mask_above_line_fraction={self.mask_above_line_fraction} (оставляем верхние {int(100*self.mask_above_line_fraction)}%%, низ отсекается)"
            )
        self.get_logger().info(
            f"Accumulation: {self.accumulate_n} frames, "
            f"min_coverage={self.min_coverage * 100:.0f}%"
        )
        if self._clahe_enable:
            self.get_logger().info(
                f"CLAHE RGB: clip_limit={self._clahe_clip_limit}, tile_grid_size={self._clahe_tile_grid_size}"
            )
        if self._debug_yolo:
            self.get_logger().info(
                "Debug YOLO: ~/debug/yolo_mask_on_rgb (PCD, с кэшем) | "
                "~/debug/yolo_mask_on_image_raw (image_raw, в реальном времени без кэша)"
            )
            if self._debug_image_raw_topic:
                self.get_logger().info(f"  image_raw topic: {self._debug_image_raw_topic}")

    def _on_image_raw(self, msg: Image) -> None:
        """YOLO по image_raw → своя маска (без ресайза), публикуем mask на image_raw."""
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            return
        if img.ndim != 3 or img.shape[2] != 3:
            return
        if self._clahe_enable:
            img = apply_clahe_rgb(img, self._clahe_clip_limit, self._clahe_tile_grid_size)
        mask_raw, yolo_result = self._segment_union_mask(img, cache_key="image_raw")
        # Наложение маски на RGB: попиксельно — где mask > 0 оставляем пиксель, иначе чёрный
        masked = np.where(mask_raw[:, :, np.newaxis] > 0, img, 0).astype(np.uint8)
        # Визуализация масок по классам с подписями (только RGB)
        if yolo_result is not None and yolo_result.masks is not None and hasattr(yolo_result, "boxes") and yolo_result.boxes is not None:
            masked = self._draw_class_masks_and_labels(masked, yolo_result, img.shape[1], img.shape[0])
        out = self.bridge.cv2_to_imgmsg(masked, encoding="rgb8")
        out.header = msg.header
        self.pub_debug_yolo_on_image_raw.publish(out)

    def _draw_class_masks_and_labels(
        self, masked_rgb: np.ndarray, yolo_result, W: int, H: int
    ) -> np.ndarray:
        """Отрисовка масок по классам и подписей на маскированном RGB (только для image_raw)."""
        vis = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR)
        m = yolo_result.masks.data
        boxes = yolo_result.boxes
        if m is None or boxes is None or len(m) == 0:
            return masked_rgb
        n = len(m)
        overlay_alpha = 0.45
        for i in range(n):
            class_id = int(boxes.cls[i].item())
            class_name = self._yolo_class_names.get(class_id, str(class_id))
            if _is_bg_class(class_name):
                continue
            mask_i = (m[i] > 0.5).to("cpu").numpy().astype(np.uint8)
            if mask_i.shape[0] != H or mask_i.shape[1] != W:
                mask_i = cv2.resize(mask_i, (W, H), interpolation=cv2.INTER_NEAREST)
            cutoff_row = int(self.mask_above_line_fraction * H) if 0.0 < self.mask_above_line_fraction < 1.0 else -1
            if cutoff_row >= 0 and not _mask_has_pixel_above_line(mask_i, H, W, cutoff_row):
                continue
            color = _CLASS_COLORS_BGR[class_id % len(_CLASS_COLORS_BGR)]
            # Полупрозрачная заливка маски классом
            where = mask_i[:, :, np.newaxis] > 0
            vis = np.where(
                where,
                (vis * (1 - overlay_alpha) + np.array(color, dtype=np.uint8) * overlay_alpha).astype(np.uint8),
                vis,
            )
            # Подпись класса у bbox
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1 = int(xyxy[0]), int(xyxy[1])
            (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, class_name, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

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

    def _segment_union_mask(self, rgb_u8: np.ndarray, cache_key: str = "pcd"):
        """Маска YOLO: 0=фон, 255=объект.

        - cache_key="pcd": маска по RGB из PointCloud, с temporal smoothing (кэш при промахах).
          Возвращает (mask, None).
        - cache_key="image_raw": маска по image_raw, без кэша. Возвращает (mask, result) для визуализации классов.
        """
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
        detected = not (r.masks is None or r.masks.data is None or len(r.masks.data) == 0)

        if detected:
            m = r.masks.data  # (N,h,w) torch, вероятности по пикселям
            valid_ix = list(range(len(m)))
            # Исключаем классы bg* из маски
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes.cls) == len(m) and self._yolo_class_names:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                valid_ix = [
                    i for i in valid_ix
                    if not _is_bg_class(self._yolo_class_names.get(cls_ids[i], str(cls_ids[i])))
                ]
            # Оставляем только экземпляры, у которых есть хотя бы один пиксель выше линии (иначе — фон, убираем класс)
            cutoff_row = int(self.mask_above_line_fraction * H) if 0.0 < self.mask_above_line_fraction < 1.0 else -1
            if valid_ix and cutoff_row >= 0:
                kept = []
                for i in valid_ix:
                    mask_np = (m[i] > 0.5).to("cpu").numpy().astype(np.uint8)
                    if _mask_has_pixel_above_line(mask_np, H, W, cutoff_row):
                        kept.append(i)
                valid_ix = kept
            if valid_ix:
                m = m[valid_ix]
            else:
                m = None
            if m is not None and len(m) > 0:
                union = (m > 0.5).any(dim=0).to("cpu").numpy().astype(np.uint8) * 255
            else:
                h, w = r.masks.data.shape[1], r.masks.data.shape[2]
                union = np.zeros((h, w), dtype=np.uint8)
            if union.shape != (H, W):
                union = cv2.resize(union, (W, H), interpolation=cv2.INTER_NEAREST)
            if cache_key == "pcd":
                self._last_valid_mask = union
                self._mask_miss_count = 0
            # Лог классов только для image_raw (без bg*, только экземпляры выше линии), не чаще раза в 2 сек
            if cache_key == "image_raw" and self._yolo_class_names and valid_ix:
                try:
                    now = time.monotonic()
                    if now - self._last_yolo_classes_log >= 2.0:
                        self._last_yolo_classes_log = now
                        ids = r.boxes.cls.cpu().numpy().astype(int)
                        names = [self._yolo_class_names.get(ids[i], str(ids[i])) for i in valid_ix]
                        unames = sorted(set(n for n in names if not _is_bg_class(n)))
                        if unames:
                            self.get_logger().info(f"YOLO детекция: классы {unames}")
                except Exception:
                    pass
            return (union, r if cache_key == "image_raw" else None)

        # YOLO ничего не нашёл
        if cache_key == "image_raw":
            return (np.zeros((H, W), dtype=np.uint8), None)

        self._mask_miss_count += 1
        if (
            self._last_valid_mask is not None
            and self._last_valid_mask.shape == (H, W)
            and self._mask_miss_count <= self.seg_mask_persist
        ):
            self.get_logger().debug(
                f"Seg miss #{self._mask_miss_count}/{self.seg_mask_persist}, reusing cached mask"
            )
            return (self._last_valid_mask, None)
        if self._mask_miss_count == self.seg_mask_persist + 1:
            self.get_logger().warning(
                f"Seg missed {self._mask_miss_count} frames in a row, mask cache expired"
            )
        return (np.zeros((H, W), dtype=np.uint8), None)

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
        if self._clahe_enable and has_rgb:
            rgb = apply_clahe_rgb(rgb, self._clahe_clip_limit, self._clahe_tile_grid_size)

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
        mask_u8, _ = self._segment_union_mask(rgb)

        # ---- debug: маска YOLO по RGB из PCD (та же логика наложения: объект=RGB, фон=0) ----
        if self._debug_yolo:
            rgb_masked = np.where(mask_u8[:, :, np.newaxis] > 0, rgb, 0).astype(np.uint8)
            img_msg = self.bridge.cv2_to_imgmsg(rgb_masked, encoding="rgb8")
            img_msg.header = msg.header
            self.pub_debug_yolo_rgb.publish(img_msg)

        # ---- build heightmap for THIS frame ----
        color_hm, height_hm, mask_hm = build_heightmaps(
            rgb_u8=rgb, xyz=xyz, spec=self.spec, mask_u8=mask_u8
        )

        # ---- merge into accumulator (fill pixels that have data) ----
        has_data = height_hm != 0
        new_pixels = int(has_data.sum())
        self._acc_color[has_data] = color_hm[has_data]
        self._acc_height[has_data] = height_hm[has_data]
        self._acc_mask[has_data] = mask_hm[has_data]
        self._acc_count += 1

        S = self.spec.size
        total_px = S * S
        filled = int(np.count_nonzero(self._acc_height))
        coverage = filled / total_px

        self.get_logger().info(
            f"Frame {self._acc_count}/{self.accumulate_n} | "
            f"this={new_pixels}px | acc={filled}/{total_px} "
            f"({coverage * 100:.1f}%)"
        )

        # ---- check if ready to publish ----
        if self._acc_count < self.accumulate_n:
            return

        # ---- publish accumulated heightmap ----
        if coverage < self.min_coverage:
            self.get_logger().warning(
                f"Coverage {coverage * 100:.1f}% < {self.min_coverage * 100:.1f}% "
                f"after {self.accumulate_n} frames — skipping, resetting"
            )
            self._reset_accumulator()
            return

        self.get_logger().info(
            f"Publishing accumulated heightmap: {filled}px "
            f"({coverage * 100:.1f}% coverage) from {self._acc_count} frames"
        )
        self._publish_heightmaps(msg.header.stamp, out_frame)
        self._reset_accumulator()

    def _reset_accumulator(self) -> None:
        self._acc_color[:] = 0
        self._acc_height[:] = 0
        self._acc_mask[:] = 0
        self._acc_count = 0

    def _publish_heightmaps(self, stamp, frame_id: str) -> None:
        color_hm = self._acc_color
        height_hm = self._acc_height
        mask_hm = self._acc_mask

        msg_color = self.bridge.cv2_to_imgmsg(color_hm, encoding="rgb8")
        msg_color.header.stamp = stamp
        msg_color.header.frame_id = frame_id
        self.pub_hm_color.publish(msg_color)

        msg_height = self.bridge.cv2_to_imgmsg(height_hm.astype(np.float32), encoding="32FC1")
        msg_height.header.stamp = stamp
        msg_height.header.frame_id = frame_id
        self.pub_hm_height.publish(msg_height)

        # height_vis
        hm = height_hm
        finite_h = np.isfinite(hm) & (hm != 0)
        vis = np.zeros_like(hm, dtype=np.uint8)
        if finite_h.any():
            vmin = float(np.nanmin(hm[finite_h]))
            vmax = float(np.nanmax(hm[finite_h]))
            if vmax > vmin:
                vis = (np.clip((hm - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0).astype(np.uint8)

        msg_vis = self.bridge.cv2_to_imgmsg(vis, encoding="mono8")
        msg_vis.header.stamp = stamp
        msg_vis.header.frame_id = frame_id
        self.pub_hm_vis.publish(msg_vis)

        mask_vis = (mask_hm > 0).astype(np.uint8) * 255
        msg_mask = self.bridge.cv2_to_imgmsg(mask_vis, encoding="mono8")
        msg_mask.header.stamp = stamp
        msg_mask.header.frame_id = frame_id
        self.pub_hm_mask.publish(msg_mask)


def main() -> None:
    rclpy.init()
    node = HeightmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
