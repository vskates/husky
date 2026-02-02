# from __future__ import annotations

# import os
# import numpy as np
# import cv2

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PoseStamped
# from cv_bridge import CvBridge

# import message_filters

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision


# # ---------------- Model ----------------

# def densenet_no_weights():
#     try:
#         return torchvision.models.densenet121(weights=None)
#     except TypeError:
#         return torchvision.models.densenet121(pretrained=False)


# class GraspNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.color_trunk = densenet_no_weights()
#         self.depth_trunk = densenet_no_weights()

#         self.head = nn.Sequential(
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2048, 64, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, 1),
#         )

#     def forward(self, color, depth):
#         c = self.color_trunk.features(color)
#         d = self.depth_trunk.features(depth)
#         x = torch.cat([c, d], dim=1)
#         x = self.head(x)
#         x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)
#         return x.squeeze(1)


# # ---------------- Node ----------------

# class GraspInferenceNode(Node):
#     def __init__(self):
#         super().__init__("grasp_inference_node")
#         self.bridge = CvBridge()

#         # ---- Parameters ----
#         self.declare_parameter("color_topic", "/heightmap_node/heightmap/color")
#         self.declare_parameter("height_topic", "/heightmap_node/heightmap/height")
#         self.declare_parameter("model_path", "")
#         self.declare_parameter("force_cpu", False)
#         self.declare_parameter("grasp_depth_offset", 0.0)
#         self.declare_parameter("sync_slop", 0.1)  # секунды для синхронизации
        
#         # Параметры heightmap (должны совпадать с heightmap node)
#         self.declare_parameter("hm_size", 224)
#         self.declare_parameter("hm_resolution", 0.002)
#         self.declare_parameter("plane_min", [-0.224, -0.224])

#         # ---- Read parameters ----
#         color_topic = self.get_parameter("color_topic").value
#         height_topic = self.get_parameter("height_topic").value
#         model_path = self.get_parameter("model_path").value
#         force_cpu = self.get_parameter("force_cpu").value
#         self.grasp_depth_offset = float(self.get_parameter("grasp_depth_offset").value)
#         sync_slop = float(self.get_parameter("sync_slop").value)
        
#         self.hm_size = int(self.get_parameter("hm_size").value)
#         self.hm_resolution = float(self.get_parameter("hm_resolution").value)
#         self.plane_min = np.array(self.get_parameter("plane_min").value, np.float32)

#         # ---- Model ----
#         use_cuda = torch.cuda.is_available() and not force_cpu
#         self.device = torch.device("cuda" if use_cuda else "cpu")
        
#         self.get_logger().info(f"Using device: {self.device}")
        
#         self.model = GraspNet().to(self.device).eval()
        
#         if model_path and os.path.exists(model_path):
#             try:
#                 self.get_logger().info(f"Loading model from: {model_path}")
#                 checkpoint = torch.load(model_path, map_location=self.device)
#                 self.model.load_state_dict(checkpoint)
#                 self.get_logger().info("✓ Model loaded successfully!")
#             except Exception as e:
#                 self.get_logger().error(f"✗ Failed to load model: {e}")
#                 self.get_logger().warn("Running with random weights!")
#         else:
#             self.get_logger().warn(f"Model file not found: {model_path}")
#             self.get_logger().warn("Running with random weights!")

#         # ---- Publishers ----
#         self.pub_q_canvas = self.create_publisher(Image, "~/q_canvas", 10)
#         self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)

#         # ---- Subscribers with synchronization ----
#         self.sub_color = message_filters.Subscriber(self, Image, color_topic)
#         self.sub_height = message_filters.Subscriber(self, Image, height_topic)
        
#         # Approximate time synchronizer
#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.sub_color, self.sub_height],
#             queue_size=10,
#             slop=sync_slop
#         )
#         self.ts.registerCallback(self._on_heightmaps)

#         self.get_logger().info(f"Subscribed to:")
#         self.get_logger().info(f"  color: {color_topic}")
#         self.get_logger().info(f"  height: {height_topic}")
#         self.get_logger().info("Waiting for synchronized heightmaps...")

#     @torch.inference_mode()
#     def _on_heightmaps(self, color_msg: Image, height_msg: Image):
#         """Callback when synchronized color and height heightmaps arrive"""
        
#         try:
#             # Convert ROS messages to numpy
#             color_hm = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
#             height_hm = self.bridge.imgmsg_to_cv2(height_msg, desired_encoding="32FC1")
            
#         except Exception as e:
#             self.get_logger().error(f"Failed to convert images: {e}")
#             return

#         # Check shapes
#         if color_hm.shape[:2] != (self.hm_size, self.hm_size):
#             self.get_logger().warn(f"Color heightmap size mismatch: {color_hm.shape[:2]} != {self.hm_size}")
#             return
        
#         if height_hm.shape != (self.hm_size, self.hm_size):
#             self.get_logger().warn(f"Height heightmap size mismatch: {height_hm.shape} != {self.hm_size}")
#             return

#         # ---- Prepare inputs for model ----
#         color_t = torch.from_numpy(color_hm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
#         depth_t = torch.from_numpy(height_hm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
#         # Handle NaN values
#         depth_t = torch.nan_to_num(depth_t, nan=0.0)

#         # Resize to model input size (448x448)
#         color_t = F.interpolate(color_t, size=(448, 448), mode="bilinear", align_corners=False)
#         depth_t = F.interpolate(depth_t, size=(448, 448), mode="bilinear", align_corners=False)

#         # Normalize
#         color_t = (color_t / 255.0 - 0.5) / 0.5  # [-1, 1]
#         depth_t = depth_t.repeat(1, 3, 1, 1)     # 3 channels

#         # ---- Model forward ----
#         q = self.model(color_t, depth_t)[0]  # (224, 224)
#         q_np = q.cpu().numpy()

#         # ---- Find best grasp ----
#         # Mask out invalid regions (where height is NaN)
#         valid = np.isfinite(height_hm)
#         q_np[~valid] = -1e9
        
#         row, col = np.unravel_index(np.argmax(q_np), q_np.shape)

#         # ---- Reconstruct 3D pose ----
#         # X (depth) from heightmap + offset
#         x = height_hm[row, col] + self.grasp_depth_offset
        
#         # Y, Z from pixel coordinates
#         # Note: heightmap indexing: row increases down, col increases right
#         # plane_min is [y_min, z_min]
#         y = self.plane_min[0] + (self.hm_size - 1 - col) * self.hm_resolution
#         z = self.plane_min[1] + (self.hm_size - 1 - row) * self.hm_resolution

#         # ---- Publish grasp pose ----
#         pose_msg = PoseStamped()
#         pose_msg.header.stamp = color_msg.header.stamp
#         pose_msg.header.frame_id = color_msg.header.frame_id
#         pose_msg.pose.position.x = float(x)
#         pose_msg.pose.position.y = float(y)
#         pose_msg.pose.position.z = float(z)
#         pose_msg.pose.orientation.w = 1.0  # No rotation (angle = 0°)
        
#         self.pub_grasp_pose.publish(pose_msg)

#         # ---- Create and publish Q-value heatmap visualization ----
#         q_norm = q_np - q_np.min()
#         if q_norm.max() > 0:
#             q_norm /= q_norm.max()

#         # Apply colormap
#         heatmap = cv2.applyColorMap((q_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
#         # Draw best grasp location
#         heatmap = cv2.circle(heatmap, (col, row), 6, (0, 0, 255), 2)
        
#         # Publish
#         q_msg = self.bridge.cv2_to_imgmsg(heatmap, encoding="bgr8")
#         q_msg.header.stamp = color_msg.header.stamp
#         q_msg.header.frame_id = color_msg.header.frame_id
#         self.pub_q_canvas.publish(q_msg)

#         self.get_logger().info(
#             f"Grasp at pixel ({row}, {col}) -> position ({x:.3f}, {y:.3f}, {z:.3f}) "
#             f"| Q-value: {q_np[row, col]:.3f}"
#         )


# def main():
#     rclpy.init()
#     node = GraspInferenceNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()


# from __future__ import annotations

# import os
# import numpy as np
# import cv2

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import PoseStamped
# from cv_bridge import CvBridge

# import message_filters

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision


# # ---------------- Model ----------------

# def densenet_no_weights():
#     try:
#         return torchvision.models.densenet121(weights=None)
#     except TypeError:
#         return torchvision.models.densenet121(pretrained=False)


# class GraspNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.color_trunk = densenet_no_weights()
#         self.depth_trunk = densenet_no_weights()

#         self.head = nn.Sequential(
#             nn.BatchNorm2d(2048),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(2048, 64, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, 1),
#         )

#     def forward(self, color, depth):
#         c = self.color_trunk.features(color)
#         d = self.depth_trunk.features(depth)
#         x = torch.cat([c, d], dim=1)
#         x = self.head(x)
#         x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)
#         return x.squeeze(1)


# # ---------------- Node ----------------

# class GraspInferenceNode(Node):
#     def __init__(self):
#         super().__init__("grasp_inference_node")
#         self.bridge = CvBridge()

#         # ---- Parameters ----
#         self.declare_parameter("color_topic", "/heightmap_node/heightmap/color")
#         self.declare_parameter("height_topic", "/heightmap_node/heightmap/height")
#         self.declare_parameter("model_path", "")
#         self.declare_parameter("force_cpu", False)
#         self.declare_parameter("grasp_depth_offset", 0.0)
#         self.declare_parameter("sync_slop", 0.1)

#         self.declare_parameter("hm_size", 224)
#         self.declare_parameter("hm_resolution", 0.002)
#         self.declare_parameter("plane_min", [-0.224, -0.224])

#         # ---- Read parameters ----
#         color_topic = self.get_parameter("color_topic").value
#         height_topic = self.get_parameter("height_topic").value
#         model_path = self.get_parameter("model_path").value
#         force_cpu = self.get_parameter("force_cpu").value
#         self.grasp_depth_offset = float(self.get_parameter("grasp_depth_offset").value)
#         sync_slop = float(self.get_parameter("sync_slop").value)

#         self.hm_size = int(self.get_parameter("hm_size").value)
#         self.hm_resolution = float(self.get_parameter("hm_resolution").value)
#         self.plane_min = np.array(self.get_parameter("plane_min").value, np.float32)

#         # ---- Model ----
#         use_cuda = torch.cuda.is_available() and not force_cpu
#         self.device = torch.device("cuda" if use_cuda else "cpu")

#         self.get_logger().info(f"Using device: {self.device}")

#         self.model = GraspNet().to(self.device).eval()

#         if model_path and os.path.exists(model_path):
#             try:
#                 self.get_logger().info(f"Loading model from: {model_path}")
#                 checkpoint = torch.load(model_path, map_location=self.device)
#                 self.model.load_state_dict(checkpoint)
#                 self.get_logger().info("✓ Model loaded successfully!")
#             except Exception as e:
#                 self.get_logger().error(f"✗ Failed to load model: {e}")
#                 self.get_logger().warn("Running with random weights!")
#         else:
#             self.get_logger().warn(f"Model file not found: {model_path}")
#             self.get_logger().warn("Running with random weights!")

#         # ---- Publishers ----
#         self.pub_q_canvas = self.create_publisher(Image, "~/q_canvas", 10)

#         self.pub_grasp_pose_raw = self.create_publisher(
#             PoseStamped, "~/grasp_pose_raw", 10
#         )
#         self.pub_grasp_pose_rot = self.create_publisher(
#             PoseStamped, "~/grasp_pose_rotated", 10
#         )
#         self.pub_grasp_pose_rot_shift = self.create_publisher(
#             PoseStamped, "~/grasp_pose_rotated_shifted", 10
#         )

#         # ---- Subscribers ----
#         self.sub_color = message_filters.Subscriber(self, Image, color_topic)
#         self.sub_height = message_filters.Subscriber(self, Image, height_topic)

#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.sub_color, self.sub_height],
#             queue_size=10,
#             slop=sync_slop
#         )
#         self.ts.registerCallback(self._on_heightmaps)

#         self.get_logger().info("Waiting for synchronized heightmaps...")

#     @torch.inference_mode()
#     def _on_heightmaps(self, color_msg: Image, height_msg: Image):

#         try:
#             color_hm = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
#             height_hm = self.bridge.imgmsg_to_cv2(height_msg, desired_encoding="32FC1")
#         except Exception as e:
#             self.get_logger().error(f"Failed to convert images: {e}")
#             return

#         if color_hm.shape[:2] != (self.hm_size, self.hm_size):
#             return
#         if height_hm.shape != (self.hm_size, self.hm_size):
#             return

#         color_t = torch.from_numpy(color_hm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
#         depth_t = torch.from_numpy(height_hm).unsqueeze(0).unsqueeze(0).float().to(self.device)
#         depth_t = torch.nan_to_num(depth_t, nan=0.0)

#         color_t = F.interpolate(color_t, size=(448, 448), mode="bilinear", align_corners=False)
#         depth_t = F.interpolate(depth_t, size=(448, 448), mode="bilinear", align_corners=False)

#         color_t = (color_t / 255.0 - 0.5) / 0.5
#         depth_t = depth_t.repeat(1, 3, 1, 1)

#         q = self.model(color_t, depth_t)[0]
#         q_np = q.cpu().numpy()

#         valid = np.isfinite(height_hm)
#         q_np[~valid] = -1e9
#         row, col = np.unravel_index(np.argmax(q_np), q_np.shape)

#         x = height_hm[row, col] + self.grasp_depth_offset
#         y = self.plane_min[0] + (self.hm_size - 1 - col) * self.hm_resolution
#         z = self.plane_min[1] + (self.hm_size - 1 - row) * self.hm_resolution

#         # ---------- RAW pose ----------
#         # ---------- RAW pose (NOW IN TCP COORDS) ----------
#         R = np.array([
#             [ 0,  1,  0],
#             [-1,  0,  0],
#             [ 0,  0,  1],
#         ], dtype=np.float32)

#         p_raw = np.array([x, y, z], dtype=np.float32)   # как модель/heightmap выдала
#         p_tcp = R @ p_raw                                # перевод в tcp

#         # доп. линейное смещение уже в tcp:
#         p_tcp[2] += 0.06   # +6см по z (вверх)
#         p_tcp[1] += 0.14   # +14см по y (назад)

#         pose_raw = PoseStamped()
#         pose_raw.header.stamp = color_msg.header.stamp
#         pose_raw.header.frame_id = "tcp"  # или ваш реальный frame id TCP (tool0/ee_link/...)
#         pose_raw.pose.position.x = float(p_tcp[0])
#         pose_raw.pose.position.y = float(p_tcp[1])
#         pose_raw.pose.position.z = float(p_tcp[2])
#         pose_raw.pose.orientation.w = 1.0

#         self.pub_grasp_pose_raw.publish(pose_raw)

#         # ---------- ROTATED pose (Z_cam -> Y_ee) ----------
#         pose_rot = PoseStamped()
#         pose_rot.header = pose_raw.header
#         pose_rot.pose.position = pose_raw.pose.position

#         pose_rot.pose.orientation.x = -0.7071
#         pose_rot.pose.orientation.y = 0.0
#         pose_rot.pose.orientation.z = 0.0
#         pose_rot.pose.orientation.w = 0.7071

#         self.pub_grasp_pose_rot.publish(pose_rot)

# # ---------- ROTATED + SHIFTED (shift = 0 for now) ----------
#         pose_rot_shift = PoseStamped()
#         pose_rot_shift.header = pose_rot.header
#         pose_rot_shift.pose = pose_rot.pose 

#         # Простейшее смещение (по глобальным координатам)
#         pose_rot_shift.pose.position.z -= 0.14  # назад по синей оси
#         pose_rot_shift.pose.position.y += 0.06  # вверх по зелёной оси

#         self.pub_grasp_pose_rot_shift.publish(pose_rot_shift)

#         # ---------- Visualization ----------
#         q_norm = q_np - q_np.min()
#         if q_norm.max() > 0:
#             q_norm /= q_norm.max()

#         heatmap = cv2.applyColorMap((q_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
#         heatmap = cv2.circle(heatmap, (col, row), 6, (0, 0, 255), 2)

#         q_msg = self.bridge.cv2_to_imgmsg(heatmap, encoding="bgr8")
#         q_msg.header = pose_raw.header
#         self.pub_q_canvas.publish(q_msg)


# def main():
#     rclpy.init()
#     node = GraspInferenceNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
from __future__ import annotations

import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters

import torch
import torch.nn.functional as F

# ---------------- Torchvision anti-download patch ----------------
# models.py в обучении создаёт DenseNet с weights='DEFAULT', что на роботе может попытаться скачать веса.
# Здесь мы перехватываем вызов и принудительно ставим weights=None / pretrained=False.
# После этого ваш state_dict всё равно перезапишет все веса.
try:
    import torchvision

    _orig_dn121 = torchvision.models.densenet.densenet121

    def _dn121_no_download(*args, **kwargs):
        # torchvision>=0.13
        if "weights" in kwargs:
            kwargs["weights"] = None
        # torchvision<0.13
        if "pretrained" in kwargs:
            kwargs["pretrained"] = False
        return _orig_dn121(*args, **kwargs)

    torchvision.models.densenet.densenet121 = _dn121_no_download
except Exception:
    # Если torchvision не импортируется здесь, будет импортирован позже внутри models.py.
    # В таком случае просто удалите этот блок или убедитесь, что torchvision доступен.
    pass


# ---------------- Import trained model (same as training) ----------------
# Если файл ноды лежит в том же python-пакете, что и models.py, используйте относительный импорт:
try:
    from .models import GraspNet
except Exception:
    # либо абсолютный импорт (если запускаете как скрипт)
    from models import GraspNet


def _strip_module_prefix(state_dict: dict) -> dict:
    """Если веса сохранены из DataParallel, ключи будут начинаться с 'module.'."""
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
        self.declare_parameter("model_path", "")
        self.declare_parameter("force_cpu", False)

        self.declare_parameter("num_rotations", 16)
        self.declare_parameter("grasp_depth_offset", 0.0)
        self.declare_parameter("sync_slop", 0.1)

        # heightmap geometry
        self.declare_parameter("hm_size", 224)
        self.declare_parameter("hm_resolution", 0.002)
        self.declare_parameter("plane_min", [-0.224, -0.224])

        # ---- Read parameters ----
        color_topic = self.get_parameter("color_topic").value
        height_topic = self.get_parameter("height_topic").value
        model_path = self.get_parameter("model_path").value
        force_cpu = bool(self.get_parameter("force_cpu").value)

        self.num_rotations = int(self.get_parameter("num_rotations").value)
        self.grasp_depth_offset = float(self.get_parameter("grasp_depth_offset").value)
        sync_slop = float(self.get_parameter("sync_slop").value)

        self.hm_size = int(self.get_parameter("hm_size").value)
        self.hm_resolution = float(self.get_parameter("hm_resolution").value)
        self.plane_min = np.array(self.get_parameter("plane_min").value, np.float32)

        # ---- Device ----
        use_cuda = torch.cuda.is_available() and not force_cpu
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # ---- Model ----
        self.get_logger().info(f"Initializing GraspNet(num_rotations={self.num_rotations}) ...")
        self.model = GraspNet(use_cuda=use_cuda, num_rotations=self.num_rotations)

        # как в Trainer: перенос на GPU при use_cuda
        self.model = self.model.to(self.device).eval()

        if model_path and os.path.exists(model_path):
            try:
                self.get_logger().info(f"Loading model state_dict from: {model_path}")
                # Попробуем максимально совместимую загрузку
                state = torch.load(model_path, map_location=self.device)
                state = _strip_module_prefix(state)
                self.model.load_state_dict(state, strict=True)
                self.get_logger().info("✓ Model loaded successfully!")
            except Exception as e:
                self.get_logger().error(f"✗ Failed to load model: {e}")
                self.get_logger().warning("Running with random weights!")
        else:
            self.get_logger().warning(f"Model file not found: {model_path}")
            self.get_logger().warning("Running with random weights!")

        # ---- Publishers ----
        self.pub_q_canvas = self.create_publisher(Image, "~/q_canvas", 10)
        self.pub_grasp_pose = self.create_publisher(PoseStamped, "~/grasp_pose", 10)

        # ---- Subscribers with synchronization ----
        self.sub_color = message_filters.Subscriber(self, Image, color_topic)
        self.sub_height = message_filters.Subscriber(self, Image, height_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_height],
            queue_size=10,
            slop=sync_slop
        )
        self.ts.registerCallback(self._on_heightmaps)

        self.get_logger().info("Subscribed to:")
        self.get_logger().info(f"  color:  {color_topic}")
        self.get_logger().info(f"  height: {height_topic}")
        self.get_logger().info("Waiting for synchronized heightmaps...")

        # constants for normalization (как в trainer.py)
        self.rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.rgb_std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.depth_mean = 0.01
        self.depth_std = 0.03

    @torch.inference_mode()
    def _on_heightmaps(self, color_msg: Image, height_msg: Image):
        # ---- Convert ROS -> numpy ----
        try:
            color_hm = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")   # (H,W,3) uint8
            height_hm = self.bridge.imgmsg_to_cv2(height_msg, desired_encoding="32FC1")  # (H,W) float32
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # ---- Validate shapes ----
        if color_hm.shape[:2] != (self.hm_size, self.hm_size):
            self.get_logger().warning(
                f"Color heightmap size mismatch: {color_hm.shape[:2]} != {(self.hm_size, self.hm_size)}"
            )
            return
        if height_hm.shape != (self.hm_size, self.hm_size):
            self.get_logger().warning(
                f"Height heightmap size mismatch: {height_hm.shape} != {(self.hm_size, self.hm_size)}"
            )
            return

        # ---- Prepare tensors (match Trainer.forward) ----
        # color: (1,3,H,W), depth: (1,1,H,W)
        color = torch.from_numpy(color_hm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        depth = torch.from_numpy(height_hm).unsqueeze(0).unsqueeze(0).float().to(self.device)
        depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # 2x resize (H*2, W*2)
        H2 = self.hm_size * 2
        color_2x = F.interpolate(color, size=(H2, H2), mode="bilinear", align_corners=False)
        depth_2x = F.interpolate(depth, size=(H2, H2), mode="bilinear", align_corners=False)

        # pad to diagonal length (multiple of 32), same as trainer.py
        diag_length = H2 * math.sqrt(2.0)
        diag_length = int(math.ceil(diag_length / 32.0) * 32.0)
        padding_width = int((diag_length - H2) / 2)

        if padding_width > 0:
            color_2x = F.pad(color_2x, (padding_width,) * 4, mode="constant", value=0.0)
            depth_2x = F.pad(depth_2x, (padding_width,) * 4, mode="constant", value=0.0)

        # normalize RGB like ImageNet (trainer.normalize_rgb)
        input_color = (color_2x / 255.0 - self.rgb_mean) / self.rgb_std

        # depth -> 3ch + normalize (trainer.normalize_depth with mean=0.01 std=0.03)
        input_depth = torch.cat([depth_2x, depth_2x, depth_2x], dim=1)
        input_depth = (input_depth - self.depth_mean) / self.depth_std

        # ---- Forward (all rotations) ----
        output_prob_list, _ = self.model(input_color, input_depth, specific_rotation=0)

        # (R,1,Hout,Wout) -> (R,Hout,Wout)
        output_prob = torch.cat(output_prob_list, dim=0).squeeze(1)

        # crop to hm_size exactly as Trainer.forward
        # output size is diag_length/2; start/end match padding_width/2 and diag_length/2 - padding_width/2
        start = int(padding_width / 2)
        end = int((diag_length / 2) - (padding_width / 2))
        grasp_pred = output_prob[:, start:end, start:end]  # (R,hm_size,hm_size)

        # ---- Mask invalid regions (NaN in original heightmap) ----
        valid = np.isfinite(height_hm)
        valid_t = torch.from_numpy(valid).to(self.device).bool().unsqueeze(0)  # (1,H,W)
        grasp_pred = grasp_pred.masked_fill(~valid_t, -1e9)

        # ---- Best (rotation,row,col) ----
        R, H, W = grasp_pred.shape
        flat_idx = torch.argmax(grasp_pred)
        rot = int(flat_idx // (H * W))
        rem = int(flat_idx % (H * W))
        row = int(rem // W)
        col = int(rem % W)

        # one heatmap for visualization: max over rotations
        q_map = grasp_pred.max(dim=0).values  # (H,W)
        q_np = q_map.detach().cpu().numpy()

        # ---- Reconstruct 3D pose (keeping your logic) ----
        x = float(height_hm[row, col] + self.grasp_depth_offset)
        y = float(self.plane_min[0] + (self.hm_size - 1 - col) * self.hm_resolution)
        z = float(self.plane_min[1] + (self.hm_size - 1 - row) * self.hm_resolution)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = color_msg.header.stamp
        pose_msg.header.frame_id = color_msg.header.frame_id
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0  # как у вас (ориентацию не меняю)
        self.pub_grasp_pose.publish(pose_msg)

        # ---- Publish heatmap canvas ----
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

        self.get_logger().info(
            f"best rot={rot}/{self.num_rotations} | pixel=({row},{col}) -> pos=({x:.3f},{y:.3f},{z:.3f})"
        )


def main(args=None):
    rclpy.init(args=args)ф
    node = GraspInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
