#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger

import tf2_ros
from cv_bridge import CvBridge


class HandEyeCalibrationNode(Node):

    def __init__(self):
        super().__init__('handeye_calibration')

        # ---------------- CONFIG ----------------
        self.base_frame = 'base_link'
        self.wrist_frame = 'wrist_3_link'
        self.camera_frame = 'camera_link'

        self.marker_length = 0.1  # meters

        # !!! ЗАПОЛНИ !!!
        self.camera_matrix = np.array([
            [600.0, 0.0, 320.0],
            [0.0, 600.0, 240.0],
            [0.0, 0.0, 1.0]
            
        ])

        self.dist_coeffs = np.zeros(5)
        # ----------------------------------------

        self.bridge = CvBridge()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_cb, 10
        )

        # Publishers
        self.status_pub = self.create_publisher(String, '/handeye/status', 10)
        self.result_pub = self.create_publisher(
            TransformStamped, '/handeye/result', 10
        )

        # Services
        self.create_service(Trigger, '/handeye/capture', self.capture_cb)
        self.create_service(Trigger, '/handeye/compute', self.compute_cb)
        self.create_service(Trigger, '/handeye/reset', self.reset_cb)

        # Data storage
        self.R_gripper2base = []
        self.t_gripper2base = []
        self.R_target2cam = []
        self.t_target2cam = []

        self.last_t_gripper = None
        self.latest_image = None

        self.publish_status("Waiting for capture")

    # -------------------------------------------------

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    # -------------------------------------------------

    def image_cb(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='bgr8'
        )

    # -------------------------------------------------

    def capture_cb(self, request, response):
        try:
            self.capture_pose()
            response.success = True
            response.message = f"Pose {len(self.R_gripper2base)} saved"
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    # -------------------------------------------------

    def reset_cb(self, request, response):
        self.R_gripper2base.clear()
        self.t_gripper2base.clear()
        self.R_target2cam.clear()
        self.t_target2cam.clear()
        self.last_t_gripper = None

        self.publish_status("Reset done. Waiting for capture")
        response.success = True
        response.message = "Reset successful"
        return response

    # -------------------------------------------------

    def capture_pose(self):
        if self.latest_image is None:
            raise RuntimeError("No image received")

        # ---------- TF: base -> wrist ----------
        tf = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.wrist_frame,
            rclpy.time.Time()
        )

        t = tf.transform.translation
        q = tf.transform.rotation

        R_bw = self.quat_to_rot([q.x, q.y, q.z, q.w])
        t_bw = np.array([t.x, t.y, t.z])

        # invert -> wrist -> base
        R_wb = R_bw.T
        t_wb = -R_wb @ t_bw

        # check movement
        if self.last_t_gripper is not None:
            if np.linalg.norm(t_wb - self.last_t_gripper) < 0.01:
                raise RuntimeError("Robot did not move enough")

        self.last_t_gripper = t_wb.copy()

        # ---------- ArUco ----------
        gray = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

        if ids is None:
            raise RuntimeError("No ArUco detected")

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs
        )

        R_tc, _ = cv2.Rodrigues(rvecs[0])
        t_tc = tvecs[0].reshape(3)

        # ---------- Store ----------
        self.R_gripper2base.append(R_wb)
        self.t_gripper2base.append(t_wb)
        self.R_target2cam.append(R_tc)
        self.t_target2cam.append(t_tc)

        n = len(self.R_gripper2base)
        self.publish_status(f"Pose {n} saved. Move robot.")

        if n >= 5:
            self.publish_status(f"Ready to compute ({n} poses)")

    # -------------------------------------------------

    def compute_cb(self, request, response):
        if len(self.R_gripper2base) < 5:
            response.success = False
            response.message = "Need at least 5 poses"
            return response

        self.publish_status("Computing hand-eye calibration...")

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base,
            self.t_gripper2base,
            self.R_target2cam,
            self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        msg = TransformStamped()
        msg.header.frame_id = self.wrist_frame
        msg.child_frame_id = self.camera_frame

        msg.transform.translation.x = float(t_cam2gripper[0])
        msg.transform.translation.y = float(t_cam2gripper[1])
        msg.transform.translation.z = float(t_cam2gripper[2])

        q = self.rot_to_quat(R_cam2gripper)
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]

        self.result_pub.publish(msg)

        self.publish_status("Calibration done")
        response.success = True
        response.message = "Hand-eye calibration completed"
        return response

    # -------------------------------------------------

    @staticmethod
    def quat_to_rot(q):
        x, y, z, w = q
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
        ])
        return R

    @staticmethod
    def rot_to_quat(R):
        qw = np.sqrt(1.0 + np.trace(R)) / 2.0
        qx = (R[2,1] - R[1,2]) / (4*qw)
        qy = (R[0,2] - R[2,0]) / (4*qw)
        qz = (R[1,0] - R[0,1]) / (4*qw)
        return [qx, qy, qz, qw]


def main():
    rclpy.init()
    node = HandEyeCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
