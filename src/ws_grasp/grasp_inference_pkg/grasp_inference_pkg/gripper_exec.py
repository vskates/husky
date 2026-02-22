from __future__ import annotations
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
import time
import math
import threading


class GripperExecNode(Node):
    def __init__(self):
        super().__init__("gripper_exec_node")
        # ===== параметры =====
        self.declare_parameter("grasp_pose_topic", "/grasp_inference_node/grasp_pose_gripper")
        self.declare_parameter("tcp_pose_topic", "/ur5_/tcp_pose_broadcaster/pose")
        self.declare_parameter("urscript_topic", "/ur5_/urscript_interface/script_command")
        self.declare_parameter("gripper_target_topic", "/gripper/target_position")
        self.declare_parameter("gripper_current_topic", "/gripper/current_position")
        self.declare_parameter("move_acceleration", 0.05)
        self.declare_parameter("move_velocity", 0.05)
        self.declare_parameter("move_radius", 0.0)
        self.declare_parameter("gripper_close_position", 100.0)
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("auto_execute", True)
        self.declare_parameter("wait_after_move", 1.0)
        self.declare_parameter("wait_after_grip", 1.5)
        self.declare_parameter("pre_grasp_offset_y", 0.08)
        self.declare_parameter("lift_height", 0.08)

        # home_position: [x, y, z, rx, ry, rz] в координатах base (для movep)
        self.declare_parameter("home_position", [0.103, -0.302, 0.306, 0.002, 2.301, -2.172])
        self.declare_parameter("left_right_delta", 0.05)
        self.declare_parameter("above_home_height", 0.15)

        # читаем параметры
        grasp_topic = self.get_parameter("grasp_pose_topic").value
        tcp_topic = self.get_parameter("tcp_pose_topic").value
        self.urscript_topic = self.get_parameter("urscript_topic").value
        gripper_target_topic = self.get_parameter("gripper_target_topic").value
        gripper_current_topic = self.get_parameter("gripper_current_topic").value

        self.move_accel = float(self.get_parameter("move_acceleration").value)
        self.move_vel = float(self.get_parameter("move_velocity").value)
        self.move_radius = float(self.get_parameter("move_radius").value)

        self.gripper_close = float(self.get_parameter("gripper_close_position").value)
        self.gripper_open = float(self.get_parameter("gripper_open_position").value)

        self.auto_execute = bool(self.get_parameter("auto_execute").value)
        self.wait_move = float(self.get_parameter("wait_after_move").value)
        self.wait_grip = float(self.get_parameter("wait_after_grip").value)
        self.pre_grasp_y = float(self.get_parameter("pre_grasp_offset_y").value)
        self.lift_height = float(self.get_parameter("lift_height").value)

        home_raw = self.get_parameter("home_position").value
        self.home_pos = [float(v) for v in home_raw]
        self.left_right_delta = float(self.get_parameter("left_right_delta").value)
        self.above_home_height = float(self.get_parameter("above_home_height").value)

        # ===== Состояние =====
        self.latest_grasp_pose: PoseStamped | None = None
        self.current_tcp_pose: PoseStamped | None = None
        self.current_gripper_pos: float | None = None
        self.is_executing = False
        self.one_shot_done = False

        # ===== Подписчики =====
        self.sub_grasp = self.create_subscription(
            PoseStamped, grasp_topic, self._on_grasp_pose, 10
        )
        self.sub_tcp = self.create_subscription(
            PoseStamped, tcp_topic, self._on_tcp_pose, 10
        )
        self.sub_gripper_current = self.create_subscription(
            Float32, gripper_current_topic, self._on_gripper_current, 10
        )

        # ===== Паблишеры =====
        self.pub_urscript = self.create_publisher(String, self.urscript_topic, 10)
        self.pub_gripper = self.create_publisher(Float32, gripper_target_topic, 10)

        self.get_logger().info("=" * 60)
        self.get_logger().info("GripperExecNode initialized")
        self.get_logger().info(
            f"  home_position: [{', '.join(f'{v:.3f}' for v in self.home_pos)}]"
        )
        self.get_logger().info(f"  lift_height: {self.lift_height} m")
        self.get_logger().info(f"  above_home_height: {self.above_home_height} m")
        self.get_logger().info(f"  left_right_delta: {self.left_right_delta}")
        self.get_logger().info(f"  pre_grasp_offset_y: {self.pre_grasp_y}")
        self.get_logger().info("Waiting for grasp pose...")

    # ===== callbacks =====

    def _on_tcp_pose(self, msg: PoseStamped):
        self.current_tcp_pose = msg

    def _on_gripper_current(self, msg: Float32):
        self.current_gripper_pos = msg.data

    def _on_grasp_pose(self, msg: PoseStamped):
        if self.one_shot_done:
            return

        self.latest_grasp_pose = msg
        self.get_logger().info("GRASP POSE RECEIVED — starting sequence")

        self.one_shot_done = True
        try:
            self.destroy_subscription(self.sub_grasp)
        except Exception:
            pass

        thread = threading.Thread(target=self.execute_grasp_sequence, daemon=True)
        thread.start()

    # ===== main sequence =====

    def execute_grasp_sequence(self):
        if self.current_tcp_pose is None:
            self.get_logger().info("Waiting for first TCP message...")
        while self.current_tcp_pose is None and rclpy.ok():
            time.sleep(0.01)

        gx = self.latest_grasp_pose.pose.position.x
        gy = self.latest_grasp_pose.pose.position.y
        gz = self.latest_grasp_pose.pose.position.z

        hx, hy, hz = self.home_pos[0], self.home_pos[1], self.home_pos[2]
        hrx, hry, hrz = self.home_pos[3], self.home_pos[4], self.home_pos[5]

        is_left = (gx - hx) > self.left_right_delta

        self.is_executing = True

        self.get_logger().info("")
        self.get_logger().info(f"  Grasp target: x={gx:.3f}, y={gy:.3f}, z={gz:.3f}")
        self.get_logger().info(f"  Home:         x={hx:.3f}, y={hy:.3f}, z={hz:.3f}")
        self.get_logger().info(f"  Side: {'LEFT (no pre-grasp)' if is_left else 'RIGHT (with pre-grasp)'}")
        self.get_logger().info("")

        try:
            # === 1. Move to home ===
            self.get_logger().info("[1/9] Moving to home_position...")
            self._move_to_xyzrpy(self.home_pos, "home")
            self._wait_until_reached_xyz(hx, hy, hz)
            self.get_logger().info("[1/9] Reached home_position")

            # === 2. Open gripper ===
            self.get_logger().info("[2/9] Opening gripper...")
            self._open_gripper()
            time.sleep(self.wait_grip)

            # === 3. Pre-grasp (only for RIGHT positions) ===
            if not is_left:
                pre_y = gy + self.pre_grasp_y
                pre = [gx, pre_y, gz, hrx, hry, hrz]
                self.get_logger().info(
                    f"[3/9] Moving to PRE-GRASP: x={gx:.3f}, y={pre_y:.3f}, z={gz:.3f}"
                )
                self._move_to_xyzrpy(pre, "pre-grasp")
                self._wait_until_reached_xyz(gx, pre_y, gz)
                self.get_logger().info("[3/9] Reached pre-grasp")
                time.sleep(3.0)
            else:
                self.get_logger().info("[3/9] LEFT side — skipping pre-grasp")

            # === 4. Move to grasp ===
            grasp = [gx, gy, gz, hrx, hry, hrz]
            self.get_logger().info(
                f"[4/9] Moving to GRASP: x={gx:.3f}, y={gy:.3f}, z={gz:.3f}"
            )
            self._move_to_xyzrpy(grasp, "grasp")
            self._wait_until_reached_xyz(gx, gy, gz)
            self.get_logger().info("[4/9] Reached grasp position")

            # === 5. Close gripper ===
            self.get_logger().info("[5/9] Closing gripper...")
            self._close_gripper()
            time.sleep(3.0)
            self.get_logger().info("[5/9] Gripper closed")

            # === 6. Lift ===
            lift_z = gz + self.lift_height
            lift = [gx, gy, lift_z, hrx, hry, hrz]
            self.get_logger().info(f"[6/9] Lifting to z={lift_z:.3f}...")
            self._move_to_xyzrpy(lift, "lift")
            self._wait_until_reached_xyz(gx, gy, lift_z)
            self.get_logger().info("[6/9] Lifted")

            # === 7. Above home ===
            above_z = hz + self.above_home_height
            above = [hx, hy, above_z, hrx, hry, hrz]
            self.get_logger().info(f"[7/9] Moving above home: z={above_z:.3f}...")
            self._move_to_xyzrpy(above, "above-home")
            self._wait_until_reached_xyz(hx, hy, above_z)
            self.get_logger().info("[7/9] Reached above-home")

            # === 8. Return to home ===
            self.get_logger().info("[8/9] Returning to home_position...")
            self._move_to_xyzrpy(self.home_pos, "home-return")
            self._wait_until_reached_xyz(hx, hy, hz)
            self.get_logger().info("[8/9] Reached home")

            # === 9. Done ===
            self.get_logger().info("")
            self.get_logger().info("[9/9] GRASP SEQUENCE COMPLETED")
            self.get_logger().info("")
            time.sleep(1.0)
            self._open_gripper()


        except Exception as e:
            self.get_logger().error(f"EXECUTION FAILED: {e}")

        finally:
            self.is_executing = False
            self.get_logger().info("State unlocked (one-shot finished)")

    # ===== movement helpers =====

    def _move_to_xyzrpy(self, pose6: list[float], label: str = "target"):
        x, y, z, rx, ry, rz = pose6
        urscript = (
            f"def my_prog():\n"
            f"  set_digital_out(1, True)\n"
            f"  movep(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}], "
            f"a={self.move_accel}, v={self.move_vel}, r={self.move_radius})\n"
            f'  textmsg("motion finished: {label}")\n'
            f"end"
        )
        msg = String()
        msg.data = urscript
        self.pub_urscript.publish(msg)
        self.get_logger().info(
            f"  URScript sent: p[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}] ({label})"
        )

    def _wait_until_reached_xyz(self, tx: float, ty: float, tz: float, pos_tol: float = 0.03):
        log_counter = 0
        while rclpy.ok():
            if self.current_tcp_pose is None:
                time.sleep(0.01)
                continue
            dx = self.current_tcp_pose.pose.position.x - tx
            dy = self.current_tcp_pose.pose.position.y - ty
            dz = self.current_tcp_pose.pose.position.z - tz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= pos_tol:
                return
            if log_counter % 100 == 0:
                self.get_logger().info(f"  waiting... dist={dist:.4f} m")
            log_counter += 1
            time.sleep(0.02)

    # ===== gripper =====

    def _close_gripper(self):
        msg = Float32()
        msg.data = self.gripper_close
        self.pub_gripper.publish(msg)
        self.get_logger().info(f"  Gripper CLOSE ({self.gripper_close})")

    def _open_gripper(self):
        msg = Float32()
        msg.data = self.gripper_open
        self.pub_gripper.publish(msg)
        self.get_logger().info(f"  Gripper OPEN ({self.gripper_open})")

    # ===== utils =====

    @staticmethod
    def _quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
        angle = 2.0 * math.acos(max(-1.0, min(1.0, qw)))
        sin_half = math.sin(angle / 2.0)
        if abs(sin_half) < 1e-9:
            return 0.0, 0.0, 0.0
        rx = qx / sin_half * angle
        ry = qy / sin_half * angle
        rz = qz / sin_half * angle
        return rx, ry, rz


def main(args=None):
    rclpy.init(args=args)
    node = GripperExecNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
