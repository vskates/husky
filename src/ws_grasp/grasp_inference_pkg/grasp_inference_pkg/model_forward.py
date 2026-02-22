from __future__ import annotations

import os
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import message_filters

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

import torch

# --- dependency used in training ---
try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise RuntimeError(
        "segmentation_models_pytorch is required (matches training). "
        "Install it in your ROS env, then restart the node."
    ) from e


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if any(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_q_state(ckpt) -> dict:
    """
    Trainer.save() in Isaac-5 base mode saves:
        {'q_func': state_dict}
    but we also accept direct state_dict.
    """
    if isinstance(ckpt, dict) and "q_func" in ckpt and isinstance(ckpt["q_func"], dict):
        return ckpt["q_func"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt
    raise ValueError("Unsupported checkpoint format. Expected dict with key 'q_func' or a state_dict.")


class GraspInferenceNode(Node):
    def __init__(self):
        super().__init__("grasp_inference_node")
        self.bridge = CvBridge()

        # ---- Topics ----
        self.declare_parameter("color_topic", "/heightmap_node/heightmap/color")
        self.declare_parameter("height_topic", "/heightmap_node/heightmap/height")
        self.declare_parameter("mask_topic", "/heightmap_node/heightmap/mask")

        # ---- Model ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("force_cpu", False)

        # ---- Geometry (must match HeightmapNode + sim CAMERA_WORKSPACE) ----
        self.declare_parameter("hm_size", 224)
        self.declare_parameter("hm_resolution", 0.001)
        self.declare_parameter("plane_min", [0.438, 0.888])
        self.declare_parameter("grasp_depth_offset", 0.00)  # == GRASP_DEPTH in sim (base mode)

        # ---- Sync: sync_slop (сек) — макс. разброс времени между color/height/mask,
        # чтобы ApproximateTimeSynchronizer объединил их в один вызов коллбэка ----
        self.declare_parameter("sync_slop", 0.1)

        # ---- Stabilization (EMA по позе закомментировано для дебага) ----
        self.declare_parameter("pose_ema_alpha", 0.4)

        # ---- TF output ----
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("transform_timeout", 1.0)
        # Преобразование из координат модели (raw) в camera_link: x_raw=z_cam, y_raw=-x_cam, z_raw=-y_cam
        self.declare_parameter("apply_model_to_camera_transform", True)

        # ---- Read parameters ----
        color_topic = self.get_parameter("color_topic").value
        height_topic = self.get_parameter("height_topic").value
        mask_topic = self.get_parameter("mask_topic").value

        model_path = self.get_parameter("model_path").value
        force_cpu = bool(self.get_parameter("force_cpu").value)

        self.hm_size = int(self.get_parameter("hm_size").value)
        self.hm_resolution = float(self.get_parameter("hm_resolution").value)
        self.plane_min = np.array(self.get_parameter("plane_min").value, np.float32)
        self.grasp_depth_offset = float(self.get_parameter("grasp_depth_offset").value)

        sync_slop = float(self.get_parameter("sync_slop").value)
        # self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)  # отключено для дебага
        self.pose_ema_alpha = 0.4  # не используется при отключённом EMA

        self.target_frame = self.get_parameter("target_frame").value
        self.transform_timeout = float(self.get_parameter("transform_timeout").value)
        self.apply_model_to_camera = bool(self.get_parameter("apply_model_to_camera_transform").value)

        # Матрица преобразования: координаты модели (raw) -> camera_link
        # x_raw = z_cam, y_raw = -x_cam, z_raw = -y_cam  =>  x_cam = -y_raw, y_cam = -z_raw, z_cam = x_raw
        self.model_to_camera_rotation = np.array([
            [ 0, -1,  0],  # x_cam = -y_raw
            [ 0,  0, -1],  # y_cam = -z_raw
            [ 1,  0,  0],  # z_cam = x_raw
        ], dtype=np.float32)

        # ---- TF2 ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Device ----
        use_cuda = torch.cuda.is_available() and not force_cpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ---- Model (same as Trainer(method='base')) ----
        # IMPORTANT: encoder_weights=None to avoid any downloads; checkpoint overwrites weights anyway.
        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=4,
            classes=1,
        ).to(self.device).eval()

        if model_path and os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=self.device)
            q_state = _strip_module_prefix(_extract_q_state(ckpt))
            self.model.load_state_dict(q_state, strict=True)
            self.get_logger().info(f"Loaded model: {model_path}")
        else:
            self.get_logger().error(f"Model not found: {model_path}")
            raise FileNotFoundError(model_path)

        # ---- Normalization (matches trainer(2).py) ----
        # resnet18 preprocessing params:
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # plus depth: mean=0.1, std=0.03
        mean = torch.tensor([0.485, 0.456, 0.406, 0.1], device=self.device).view(1, 4, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225, 0.03], device=self.device).view(1, 4, 1, 1)
        self._norm_mean = mean
        self._norm_std = std

        # ---- Publishers ----
        self.pub_q_canvas = self.create_publisher(Image, "~/q_canvas", 10)
        self.pub_q_map_raw = self.create_publisher(Image, "~/q_map_raw", 10)
        self.pub_grasp_pose_camera = self.create_publisher(PoseStamped, "~/grasp_pose", 10)   # camera_link
        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, "~/grasp_pose_base", 10)
        self.pub_grasp_pose_gripper = self.create_publisher(PoseStamped, "~/grasp_pose_gripper", 10)
        self.pub_marker_camera = self.create_publisher(Marker, "~/grasp_marker_camera", 10)    # синий в camera_link
        self.pub_marker_base = self.create_publisher(Marker, "~/grasp_marker", 10)           # красный в base

        # ---- Subscribers (sync 3 topics) ----
        self.sub_color = message_filters.Subscriber(self, Image, color_topic)
        self.sub_height = message_filters.Subscriber(self, Image, height_topic)
        self.sub_mask = message_filters.Subscriber(self, Image, mask_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_height, self.sub_mask],
            queue_size=10,
            slop=sync_slop,
        )
        self.ts.registerCallback(self._on_heightmaps)

        # ---- EMA state (отключено для дебага: поза без сглаживания) ----
        # self._ema_pos: np.ndarray | None = None
        self._ema_pos = None

        self.get_logger().info(f"Subscribed to: {color_topic} | {height_topic} | {mask_topic}")
        self.get_logger().info(
            f"hm_size={self.hm_size} res={self.hm_resolution} plane_min={self.plane_min.tolist()} "
            f"grasp_depth_offset={self.grasp_depth_offset}"
        )
        # self.get_logger().info(f"stabilization: pose_ema_alpha={self.pose_ema_alpha}")  # отключено для дебага
        self.get_logger().info(
            f"target_frame={self.target_frame} | apply_model_to_camera={self.apply_model_to_camera}"
        )

    def _preprocess(self, color_hm_rgb8: np.ndarray, height_hm: np.ndarray) -> torch.Tensor:
        """
        Matches Trainer.forward() for base method:
            color -> float/255, CHW
            x = concat(color, depth[None,None])
            normalize with mean/std (RGB imagenet + depth mean=0.1 std=0.03)
        """
        color = torch.from_numpy(color_hm_rgb8).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        depth = torch.from_numpy(height_hm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.cat([color, depth], dim=1)  # (1,4,H,W)
        x = (x - self._norm_mean) / self._norm_std
        return x

    @torch.inference_mode()
    def _on_heightmaps(self, color_msg: Image, height_msg: Image, mask_msg: Image):
        color_hm = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        height_hm = self.bridge.imgmsg_to_cv2(height_msg, desired_encoding="32FC1")
        mask_hm = self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding="mono8")

        if color_hm.shape[:2] != (self.hm_size, self.hm_size):
            return
        if height_hm.shape != (self.hm_size, self.hm_size):
            return
        if mask_hm.shape != (self.hm_size, self.hm_size):
            return


        trick_hhm = height_hm / 2 + 0.24

        x = self._preprocess(color_hm, trick_hhm)

        # Unet outputs logits -> sigmoid -> q_map in [0,1]
        logits = self.model(x)            # (1,1,H,W)
        q = torch.sigmoid(logits)[0, 0]   # (H,W)

        q_np = q.detach().cpu().numpy()

        # ----- q_map_raw: карта Q ДО наложения маски (полная карта от модели) -----
        q_uint8 = (np.clip(q_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        q_colored = cv2.applyColorMap(q_uint8, cv2.COLORMAP_JET)
        q_raw_msg = self.bridge.cv2_to_imgmsg(q_colored, encoding="bgr8")
        q_raw_msg.header.stamp = color_msg.header.stamp
        q_raw_msg.header.frame_id = color_msg.header.frame_id
        self.pub_q_map_raw.publish(q_raw_msg)

        # ----- Наложение маски на Q: обнуляем (отсекаем) ячейки, где mask_hm == 0 -----
        # Для depth-пайплайна (вариант 2): подписать mask_topic на heightmap_depth/mask (YOLO on image_raw, проекция через наш xyz)
        print(f"HEIGHT MAP {height_hm[223,112]}")

        valid = np.isfinite(height_hm) & (height_hm != 0)&(height_hm<1)
        valid[:75, :] = False
        obj = (mask_hm.astype(np.uint8) > 0)
        # if not obj.any():
        obj = np.ones_like(obj, dtype=bool)
        keep = valid & obj
        if not keep.any():
            return

        q_np_masked = q_np.copy()
        q_np_masked[~keep] = -1e9

        flat_idx = int(np.argmax(q_np_masked))
        row = flat_idx // self.hm_size
        col = flat_idx % self.hm_size

        # ===== POSE RECONSTRUCTION =====
        x_raw = float(height_hm[row, col] + self.grasp_depth_offset)
        y_raw = float(self.plane_min[0] + (self.hm_size - 1 - col) * self.hm_resolution)
        z_raw = float(self.plane_min[1] + (self.hm_size - 1 - row) * self.hm_resolution)

        if self.apply_model_to_camera:
            pos_vec = np.array([x_raw, y_raw, z_raw], dtype=np.float32)
            pos_camera = self.model_to_camera_rotation @ pos_vec
            x, y, z = float(pos_camera[0]), float(pos_camera[1]), float(pos_camera[2])
        else:
            x, y, z = x_raw, y_raw, z_raw

        self.get_logger().info(
            f"RAW=({x_raw:.3f},{y_raw:.3f},{z_raw:.3f}) "
            f"CAM=({x:.3f},{y:.3f},{z:.3f}) "
            f"pixel=({row},{col})"
        )

        pose_camera = PoseStamped()
        pose_camera.header.stamp = color_msg.header.stamp
        pose_camera.header.frame_id = color_msg.header.frame_id
        pose_camera.pose.position.x = x
        pose_camera.pose.position.y = y
        pose_camera.pose.position.z = z
        pose_camera.pose.orientation.w = 1.0
        self.pub_grasp_pose_camera.publish(pose_camera)

        # ===== TF: camera_link -> base =====
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                pose_camera.header.frame_id,
                Time(),
                timeout=Duration(seconds=self.transform_timeout),
            )
            pose_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_camera, transform)
            self.pub_grasp_pose_base.publish(pose_base)

            pose_gripper = PoseStamped()
            pose_gripper.header = pose_base.header
            pose_gripper.pose.position.x = pose_base.pose.position.x + (-0.096)
            pose_gripper.pose.position.y = pose_base.pose.position.y + 0.008
            pose_gripper.pose.position.z = pose_base.pose.position.z + (-0.12)
            pose_gripper.pose.orientation.x = pose_base.pose.orientation.x
            pose_gripper.pose.orientation.y = pose_base.pose.orientation.y
            pose_gripper.pose.orientation.z = pose_base.pose.orientation.z
            pose_gripper.pose.orientation.w = pose_base.pose.orientation.w
            self.pub_grasp_pose_gripper.publish(pose_gripper)

            self.get_logger().info(
                f"[TF] tcp:     x={pose_base.pose.position.x:.3f} "
                f"y={pose_base.pose.position.y:.3f} z={pose_base.pose.position.z:.3f}\n"
                f"[TF] gripper: x={pose_gripper.pose.position.x:.3f} "
                f"y={pose_gripper.pose.position.y:.3f} z={pose_gripper.pose.position.z:.3f}"
            )
        except Exception as e:
            self.get_logger().error(f"TF transform failed: {e}")

        # ===== Q canvas: карта Q ПОСЛЕ маски (только объект) + точка грипа =====
        q_vis = q_np.copy()
        q_vis[~keep] = np.nan
        vmin = np.nanmin(q_vis)
        vmax = np.nanmax(q_vis)
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            q_norm = (q_vis - vmin) / (vmax - vmin)
        else:
            q_norm = np.zeros_like(q_np, dtype=np.float32)

        q_img = (np.nan_to_num(q_norm, nan=0.0) * 255.0).astype(np.uint8)
        heatmap = cv2.applyColorMap(q_img, cv2.COLORMAP_JET)
        heatmap = cv2.circle(heatmap, (col, row), 6, (0, 0, 255), 2)

        q_msg = self.bridge.cv2_to_imgmsg(heatmap, encoding="bgr8")
        q_msg.header.stamp = color_msg.header.stamp
        q_msg.header.frame_id = color_msg.header.frame_id
        self.pub_q_canvas.publish(q_msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraspInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
