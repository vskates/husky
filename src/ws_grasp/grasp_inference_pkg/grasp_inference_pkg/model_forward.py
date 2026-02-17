from __future__ import annotations

import os
import math
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
import torch.nn.functional as F

# anti-download patch
try:
    import torchvision
    _orig_dn121 = torchvision.models.densenet.densenet121

    def _dn121_no_download(*args, **kwargs):
        if "weights" in kwargs:
            kwargs["weights"] = None
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False
        return _orig_dn121(*args, **kwargs)

    torchvision.models.densenet.densenet121 = _dn121_no_download
except Exception:
    pass

try:
    from .models import GraspNet
except Exception:
    from models import GraspNet


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


class GraspInferenceNode(Node):
    def __init__(self):
        super().__init__("grasp_inference_node")
        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter("color_topic", "/heightmap_node/heightmap/color")
        self.declare_parameter("height_topic", "/heightmap_node/heightmap/height")
        self.declare_parameter("mask_topic", "/heightmap_node/heightmap/mask")

        self.declare_parameter("model_path", "")
        self.declare_parameter("force_cpu", False)

        self.declare_parameter("num_rotations", 16)
        self.declare_parameter("grasp_depth_offset", 0.0)
        self.declare_parameter("sync_slop", 0.1)

        self.declare_parameter("hm_size", 224)
        self.declare_parameter("hm_resolution", 0.002)
        self.declare_parameter("plane_min", [-0.224, -0.224])
        
        # Стабилизация Q-map
        self.declare_parameter("q_blur_sigma", 3.0)
        self.declare_parameter("pose_ema_alpha", 0.4)

        # Трансформация
        self.declare_parameter("target_frame", "base")
        self.declare_parameter("transform_timeout", 1.0)
        self.declare_parameter("apply_model_to_camera_transform", True)

        # ---- Read parameters ----
        color_topic = self.get_parameter("color_topic").value
        height_topic = self.get_parameter("height_topic").value
        mask_topic = self.get_parameter("mask_topic").value

        model_path = self.get_parameter("model_path").value
        force_cpu = bool(self.get_parameter("force_cpu").value)

        self.num_rotations = int(self.get_parameter("num_rotations").value)
        self.grasp_depth_offset = float(self.get_parameter("grasp_depth_offset").value)
        sync_slop = float(self.get_parameter("sync_slop").value)

        self.hm_size = int(self.get_parameter("hm_size").value)
        self.hm_resolution = float(self.get_parameter("hm_resolution").value)
        self.plane_min = np.array(self.get_parameter("plane_min").value, np.float32)
        
        self.q_blur_sigma = float(self.get_parameter("q_blur_sigma").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)

        self.target_frame = self.get_parameter("target_frame").value
        self.transform_timeout = float(self.get_parameter("transform_timeout").value)
        self.apply_model_to_camera = bool(self.get_parameter("apply_model_to_camera_transform").value)

        # Матрица преобразования из координат модели (raw) в реальный camera_link
        # Слева raw, справа real:
        # x_raw = z_real  →  z_real = x_raw
        # y_raw = -x_real →  x_real = -y_raw
        # z_raw = -y_real →  y_real = -z_raw
        self.model_to_camera_rotation = np.array([
            [ 0, -1,  0],  # x_real = -y_raw
            [ 0,  0, -1],  # y_real = z_raw
            [ 1,  0,  0],  # z_real = x_raw
        ], dtype=np.float32)

        # ---- TF2 Buffer & Listener ----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- Device ----
        use_cuda = torch.cuda.is_available() and not force_cpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ---- Model ----
        self.model = GraspNet(use_cuda=use_cuda, num_rotations=self.num_rotations).to(self.device).eval()

        if model_path and os.path.exists(model_path):
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(_strip_module_prefix(state), strict=True)
            self.get_logger().info(f"✓ Loaded model: {model_path}")
        else:
            self.get_logger().warning(f"Model not found: {model_path} (running with random weights)")

        # ---- Publishers ----
        self.pub_q_canvas = self.create_publisher(Image, "~/q_canvas", 10)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)
        self.pub_grasp_pose_base = self.create_publisher(PoseStamped, "~/grasp_pose_base", 10)
        self.pub_grasp_pose_gripper = self.create_publisher(PoseStamped, "~/grasp_pose_gripper", 10)
        self.pub_grasp_marker = self.create_publisher(Marker, "~/grasp_marker", 10)
        self.pub_grasp_marker_camera = self.create_publisher(Marker, "~/grasp_marker_camera", 10)

        # ---- Subscribers with sync (3 topics) ----
        self.sub_color = message_filters.Subscriber(self, Image, color_topic)
        self.sub_height = message_filters.Subscriber(self, Image, height_topic)
        self.sub_mask = message_filters.Subscriber(self, Image, mask_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_height, self.sub_mask],
            queue_size=10,
            slop=sync_slop
        )
        self.ts.registerCallback(self._on_heightmaps)

        # constants
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.rgb_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.depth_mean = 0.01
        self.depth_std = 0.03

        # ---- EMA state for pose smoothing ----
        self._ema_pos: np.ndarray | None = None

        self.get_logger().info("Subscribed to:")
        self.get_logger().info(f"  color:  {color_topic}")
        self.get_logger().info(f"  height: {height_topic}")
        self.get_logger().info(f"  mask:   {mask_topic}")
        self.get_logger().info(f"Target frame: {self.target_frame}")
        self.get_logger().info(f"Apply model→camera transform: {self.apply_model_to_camera}")
        self.get_logger().info(
            f"Q stabilization: blur_sigma={self.q_blur_sigma}, "
            f"pose_ema_alpha={self.pose_ema_alpha}"
        )

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

        # tensors
        color = torch.from_numpy(color_hm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        depth = torch.from_numpy(height_hm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # resize 2x
        H2 = self.hm_size * 2
        color_2x = F.interpolate(color, size=(H2, H2), mode="bilinear", align_corners=False)
        depth_2x = F.interpolate(depth, size=(H2, H2), mode="bilinear", align_corners=False)

        # pad to diagonal length
        diag_length = H2 * math.sqrt(2.0)
        diag_length = int(math.ceil(diag_length / 32.0) * 32.0)
        padding_width = int((diag_length - H2) / 2)
        if padding_width > 0:
            color_2x = F.pad(color_2x, (padding_width,) * 4, mode="constant", value=0.0)
            depth_2x = F.pad(depth_2x, (padding_width,) * 4, mode="constant", value=0.0)

        input_color = (color_2x / 255.0 - self.rgb_mean) / self.rgb_std
        input_depth = torch.cat([depth_2x, depth_2x, depth_2x], dim=1)
        input_depth = (input_depth - self.depth_mean) / self.depth_std

        # forward
        output_prob_list, _ = self.model(input_color, input_depth, specific_rotation=0)
        output_prob = torch.cat(output_prob_list, dim=0).squeeze(1)

        start = int(padding_width / 2)
        end = int((diag_length / 2) - (padding_width / 2))
        grasp_pred = output_prob[:, start:end, start:end]

        # valid mask
        valid = np.isfinite(height_hm)
        obj = (mask_hm.astype(np.uint8) > 0)
        if not obj.any():
            obj = np.ones_like(obj, dtype=bool)

        keep = (valid & obj)
        keep_t = torch.from_numpy(keep).to(self.device).bool().unsqueeze(0)
        grasp_pred = grasp_pred.masked_fill(~keep_t, -1e9)

        # ---- Gaussian blur on Q-map for stable argmax ----
        if self.q_blur_sigma > 0:
            q_for_argmax = grasp_pred.detach().cpu().numpy()
            for i in range(q_for_argmax.shape[0]):
                q_slice = q_for_argmax[i]
                valid_q = q_slice > -1e8
                if valid_q.any():
                    blurred = cv2.GaussianBlur(
                        q_slice, ksize=(0, 0), sigmaX=self.q_blur_sigma,
                    )
                    q_slice[valid_q] = blurred[valid_q]
                q_for_argmax[i] = q_slice
            grasp_smooth = torch.from_numpy(q_for_argmax).to(self.device)
        else:
            grasp_smooth = grasp_pred

        # argmax on smoothed Q
        R, H, W = grasp_smooth.shape
        flat_idx = torch.argmax(grasp_smooth)
        rot = int(flat_idx // (H * W))
        rem = int(flat_idx % (H * W))
        row = int(rem // W)
        col = int(rem % W)

        q_map = grasp_pred.max(dim=0).values
        q_np = q_map.detach().cpu().numpy()

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

        # ---- EMA smoothing on position ----
        new_pos = np.array([x, y, z], dtype=np.float64)
        if self._ema_pos is None:
            self._ema_pos = new_pos
        else:
            a = self.pose_ema_alpha
            self._ema_pos = a * new_pos + (1.0 - a) * self._ema_pos

        x, y, z = float(self._ema_pos[0]), float(self._ema_pos[1]), float(self._ema_pos[2])

        self.get_logger().info(
            f"RAW=({x_raw:.3f},{y_raw:.3f},{z_raw:.3f}) "
            f"EMA=({x:.3f},{y:.3f},{z:.3f}) "
            f"pixel=({row},{col})"
        )

        # Создаем PoseStamped в camera_link
        pose_camera = PoseStamped()
        pose_camera.header.stamp = color_msg.header.stamp
        pose_camera.header.frame_id = color_msg.header.frame_id
        pose_camera.pose.position.x = x
        pose_camera.pose.position.y = y
        pose_camera.pose.position.z = z
        pose_camera.pose.orientation.w = 1.0
        
        # Публикуем в camera_link
        self.pub_grasp_pose.publish(pose_camera)
        
        # Маркер в camera_link (СИНИЙ)
        marker_camera = Marker()
        marker_camera.header = pose_camera.header
        marker_camera.ns = "grasp_point_camera"
        marker_camera.id = 1
        marker_camera.type = Marker.SPHERE
        marker_camera.action = Marker.ADD
        marker_camera.pose = pose_camera.pose
        marker_camera.scale.x = 0.05
        marker_camera.scale.y = 0.05
        marker_camera.scale.z = 0.05
        marker_camera.color.r = 0.0
        marker_camera.color.g = 0.0
        marker_camera.color.b = 1.0
        marker_camera.color.a = 1.0
        marker_camera.lifetime = Duration(seconds=2.0).to_msg()
        self.pub_grasp_marker_camera.publish(marker_camera)

        # ===== ТРАНСФОРМАЦИЯ в base_link =====
        try:
            # Используем последнюю доступную трансформацию
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                pose_camera.header.frame_id,
                Time(),  # последняя доступная
                timeout=Duration(seconds=self.transform_timeout)
            )
            
            # Применяем трансформацию
            pose_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_camera, transform)
            
            # Публикуем преобразованную позу
            self.pub_grasp_pose_base.publish(pose_base)
            


            #Covnvert from tool0_controlller(tcp) to real gripper
            pose_gripper = PoseStamped()
            pose_gripper.header = pose_base.header

            pose_gripper.pose.position.x = pose_base.pose.position.x
            pose_gripper.pose.position.y = pose_base.pose.position.y
            pose_gripper.pose.position.z = pose_base.pose.position.z

            pose_gripper.pose.orientation.x = pose_base.pose.orientation.x
            pose_gripper.pose.orientation.y = pose_base.pose.orientation.y
            pose_gripper.pose.orientation.z = pose_base.pose.orientation.z
            pose_gripper.pose.orientation.w = pose_base.pose.orientation.w


            # смещение в системе base (!)
            pose_gripper.pose.position.x += -0.026
            pose_gripper.pose.position.z += -0.12
            pose_gripper.pose.position.y += 0.008



            # публикуем доп. позу
            self.pub_grasp_pose_gripper.publish(pose_gripper)

            # лог
            self.get_logger().info(
                f"[TF] tcp:     x={pose_base.pose.position.x:.3f} "
                f"y={pose_base.pose.position.y:.3f} z={pose_base.pose.position.z:.3f}\n"
                f"[TF] gripper: x={pose_gripper.pose.position.x:.3f} "
                f"y={pose_gripper.pose.position.y:.3f} z={pose_gripper.pose.position.z:.3f}"
            )



            
            # ===== ВИЗУАЛИЗАЦИЯ МАРКЕРА в base_link =====
            marker = Marker()
            marker.header = pose_base.header
            marker.ns = "grasp_point"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Позиция
            marker.pose = pose_base.pose
            
            # Размер (5см сфера)
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            # Цвет (КРАСНЫЙ - после трансформации)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            # Время жизни
            marker.lifetime = Duration(seconds=2.0).to_msg()
            
            self.pub_grasp_marker.publish(marker)
            
            self.get_logger().info(
                f"🎯 best rot={rot}/{self.num_rotations} | pixel=({row},{col})\n"
                f"   RAW model: x={x_raw:.3f}, y={y_raw:.3f}, z={z_raw:.3f}\n"
                f"   {pose_camera.header.frame_id}: x={x:.3f}, y={y:.3f}, z={z:.3f}\n"
                f"   {self.target_frame}: x={pose_base.pose.position.x:.3f}, "
                f"y={pose_base.pose.position.y:.3f}, z={pose_base.pose.position.z:.3f}\n"
                f"   Transform: tx={transform.transform.translation.x:.3f}, "
                f"ty={transform.transform.translation.y:.3f}, tz={transform.transform.translation.z:.3f}"
            )
            
        except Exception as e:
            self.get_logger().error(f"Failed to transform pose: {e}")

        # ===== Q CANVAS =====
        q_norm = q_np - np.nanmin(q_np)
        q_max = np.nanmax(q_norm)
        if q_max > 1e-12:
            q_norm = q_norm / q_max
        q_img = (np.clip(q_norm, 0.0, 1.0) * 255.0).astype(np.uint8)

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