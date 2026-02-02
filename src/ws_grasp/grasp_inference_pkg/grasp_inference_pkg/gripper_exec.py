from __future__ import annotations
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32
import time
import math
import threading  # <-- для выполнения grasp_sequence в отдельном потоке


class GripperExecNode(Node):
    def __init__(self):
        super().__init__("gripper_exec_node")
        # ===== параметры =====
        self.declare_parameter("grasp_pose_topic", "/model_forward/grasp_pose_gripper")
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
        self.declare_parameter("wait_after_move", 2.0)
        self.declare_parameter("wait_after_grip", 1.5)
        self.declare_parameter("pre_grasp_offset_y", 0.05)
        self.declare_parameter("lift_height", 0.05)  # <-- НОВЫЙ ПАРАМЕТР: высота подъема (5 см)

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
        self.lift_height = float(self.get_parameter("lift_height").value)  # <-- НОВЫЙ ПАРАМЕТР

        # ===== Состояние =====
        self.latest_grasp_pose: PoseStamped | None = None
        self.frozen_grasp_pose: PoseStamped | None = None
        self.frozen_initial_tcp: PoseStamped | None = None
        self.current_tcp_pose: PoseStamped | None = None
        self.current_gripper_pos: float | None = None
        self.is_executing = False
        self.one_shot_done = False  # обязательно инициализируем

        # ===== Подписчики =====
        self.sub_grasp = self.create_subscription(
            PoseStamped,
            grasp_topic,
            self._on_grasp_pose,
            10
        )

        self.sub_tcp = self.create_subscription(
            PoseStamped,
            tcp_topic,
            self._on_tcp_pose,
            10
        )

        self.sub_gripper_current = self.create_subscription(
            Float32,
            gripper_current_topic,
            self._on_gripper_current,
            10
        )

        # ===== Паблишеры =====
        self.pub_urscript = self.create_publisher(String, self.urscript_topic, 10)
        self.pub_gripper = self.create_publisher(Float32, gripper_target_topic, 10)

        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 GripperExecNode initialized")
        self.get_logger().info(f"⬆️  Lift height after grasp: {self.lift_height} m")
        self.get_logger().info("⏳ Waiting for grasp pose...")

    def _on_tcp_pose(self, msg: PoseStamped):
        """Обновляем текущую TCP (callback работает в spin)"""
        self.current_tcp_pose = msg

    def _on_gripper_current(self, msg: Float32):
        self.current_gripper_pos = msg.data

    def _on_grasp_pose(self, msg: PoseStamped):
        """
        При первой пришедшей grasp_pose — запускаем одноразовый execution в отдельном потоке.
        Последующие позы игнорируем.
        """
        if self.one_shot_done:
            self.get_logger().warning("⚠️ New grasp pose received but IGNORED (one-shot done)")
            return

        self.latest_grasp_pose = msg
        self.get_logger().info("📍 FIRST GRASP POSE RECEIVED")

        self.one_shot_done = True
        try:
            self.destroy_subscription(self.sub_grasp)
            self.get_logger().info("🔕 Grasp subscriber destroyed (one-shot mode)")
        except Exception:
            pass

        # запускаем execution в отдельном потоке, spin продолжает обновлять TCP
        thread = threading.Thread(target=self.execute_grasp_sequence, daemon=True)
        thread.start()

    def execute_grasp_sequence(self):
        """
        1) Freeze: фиксируем grasp_pose и текущий TCP (initial)
        2) Ждём, пока текущий TCP совпадёт с target (точность 0.1 mm по каждой координате)
        3) Закрываем gripper (3 сек)
        4) НОВОЕ: Поднимаем объект на lift_height вверх (по оси Z)
        5) Возвращаемся в frozen_initial_tcp
        """
        # Ждём TCP перед началом
        if self.current_tcp_pose is None:
            self.get_logger().info("⏳ Waiting for first TCP message before starting execution...")
        while self.current_tcp_pose is None and rclpy.ok():
            time.sleep(0.01)

        # ===== Фиксация состояний =====
        self.frozen_initial_tcp = PoseStamped()
        self.frozen_initial_tcp.header = self.current_tcp_pose.header
        self.frozen_initial_tcp.pose = self.current_tcp_pose.pose

        self.frozen_grasp_pose = PoseStamped()
        self.frozen_grasp_pose.header = self.latest_grasp_pose.header
        self.frozen_grasp_pose.pose = self.latest_grasp_pose.pose

        self.is_executing = True

        self.get_logger().info("")
        self.get_logger().info("🔒 FROZEN STATE:")
        self.get_logger().info(
            f"  Grasp pose: x={self.frozen_grasp_pose.pose.position.x:.3f}, "
            f"y={self.frozen_grasp_pose.pose.position.y:.3f}, "
            f"z={self.frozen_grasp_pose.pose.position.z:.3f}"
        )
        self.get_logger().info(
            f"  Initial TCP: x={self.frozen_initial_tcp.pose.position.x:.3f}, "
            f"y={self.frozen_initial_tcp.pose.position.y:.3f}, "
            f"z={self.frozen_initial_tcp.pose.position.z:.3f}"
        )
        self.get_logger().info("")
        self.get_logger().info("🚀 STARTING GRASP EXECUTION SEQUENCE (one-shot)...")
        self.get_logger().info("")

        try:
            # ===== ШАГ 1: движение к grasp =====
            self.get_logger().info("→ Sending URScript to move to GRASP target...")
            self._move_to_pose(self.frozen_grasp_pose, "grasp-target")

            # ===== ШАГ 2: ждём совпадения TCP с target =====
            tol = 0.0001  # 0.1 мм
            self.get_logger().info(f"⏳ Waiting until TCP matches grasp target (tol={tol} m per axis)...")
            self._wait_for_tcp_match(self.frozen_grasp_pose, tol)

            # ===== ШАГ 3: захват gripper =====
            self.get_logger().info("→ Gripper: closing for 3 seconds...")
            self._close_gripper()
            time.sleep(3.0)
            self.get_logger().info("✓ Gripper: close phase complete")

            # ===== ШАГ 4: НОВОЕ - Подъем объекта вверх =====
            self.get_logger().info(f"→ Lifting object UP by {self.lift_height} m...")
            lifted_pose = self._create_lifted_pose(self.frozen_grasp_pose, self.lift_height)
            self._move_to_pose(lifted_pose, "lift-up")
            self.get_logger().info("⏳ Waiting until TCP reaches lifted position...")
            self._wait_until_reached(lifted_pose, pos_tol=0.0005)
            self.get_logger().info(f"✓ Object lifted to z={lifted_pose.pose.position.z:.3f} m")

            # ===== ШАГ 5: возврат в initial TCP =====
            self.get_logger().info("→ Returning to frozen initial TCP...")
            self._move_to_pose(self.frozen_initial_tcp, "return-to-initial")
            self.get_logger().info("⏳ Waiting until TCP returns to initial pose...")
            self._wait_until_reached(self.frozen_initial_tcp, pos_tol=0.0005)
            self.get_logger().info("✓ Returned to initial TCP")

            self.get_logger().info("")
            self.get_logger().info("🎉 GRASP SEQUENCE COMPLETED SUCCESSFULLY!")
            self.get_logger().info("")

        except Exception as e:
            self.get_logger().error(f"❌ EXECUTION FAILED: {e}")

        finally:
            # очистка
            self.frozen_grasp_pose = None
            self.frozen_initial_tcp = None
            self.is_executing = False
            self.get_logger().info("🔓 State unlocked, ready (one-shot finished)")

    def _create_lifted_pose(self, base_pose: PoseStamped, lift_height: float) -> PoseStamped:
        """
        Создает новую позу, смещенную вверх по оси Z на lift_height метров
        относительно base_pose. Ориентация остается неизменной.
        """
        lifted_pose = PoseStamped()
        lifted_pose.header = base_pose.header
        
        # Копируем позицию и добавляем смещение по Z
        lifted_pose.pose.position.x = base_pose.pose.position.x
        lifted_pose.pose.position.y = base_pose.pose.position.y
        lifted_pose.pose.position.z = base_pose.pose.position.z + lift_height
        
        # Ориентация остается такой же
        lifted_pose.pose.orientation = base_pose.pose.orientation
        
        return lifted_pose

    def _wait_for_tcp_match(self, target_pose: PoseStamped, tol: float):
        """Ждём, пока current_tcp_pose не совпадёт с target_pose (по каждой оси)"""
        log_counter = 0
        while rclpy.ok():
            if self.current_tcp_pose is None:
                if log_counter % 50 == 0:
                    self.get_logger().info("  ⏳ waiting for TCP message...")
                log_counter += 1
                time.sleep(0.02)
                continue

            dx = abs(self.current_tcp_pose.pose.position.x - target_pose.pose.position.x)
            dy = abs(self.current_tcp_pose.pose.position.y - target_pose.pose.position.y)
            dz = abs(self.current_tcp_pose.pose.position.z - target_pose.pose.position.z)

            if dx <= tol and dy <= tol and dz <= tol:
                self.get_logger().info(f"  ✅ Current TCP matches target (dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f})")
                return

            if log_counter % 50 == 0:
                self.get_logger().info(f"  ⏳ TCP not matched yet (dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f})")
            log_counter += 1
            time.sleep(0.01)

    def _wait_until_reached(self, target_pose: PoseStamped, pos_tol=0.001):
        """Ждём приближения TCP к target_pose (евклид)"""
        while rclpy.ok():
            if self.current_tcp_pose is None:
                time.sleep(0.01)
                continue
            dx = self.current_tcp_pose.pose.position.x - target_pose.pose.position.x
            dy = self.current_tcp_pose.pose.position.y - target_pose.pose.position.y
            dz = self.current_tcp_pose.pose.position.z - target_pose.pose.position.z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist <= pos_tol:
                return
            time.sleep(0.02)

    # ===== методы движения и gripper =====
    def _move_to_pose(self, target_pose: PoseStamped, label: str = "target"):
        x = target_pose.pose.position.x
        y = target_pose.pose.position.y
        z = target_pose.pose.position.z

        qx = target_pose.pose.orientation.x
        qy = target_pose.pose.orientation.y
        qz = target_pose.pose.orientation.z
        qw = target_pose.pose.orientation.w

        rx, ry, rz = self._quat_to_rotvec(qx, qy, qz, qw)

        urscript = f"""def my_prog():
  set_digital_out(1, True)
  movep(p[{x:.6f}, {y:.6f}, {z:.6f}, {rx:.6f}, {ry:.6f}, {rz:.6f}], a={self.move_accel}, v={self.move_vel}, r={self.move_radius})
  textmsg("motion finished: {label}")
end"""
        msg = String()
        msg.data = urscript
        self.pub_urscript.publish(msg)
        self.get_logger().info(
            f"  📤 URScript sent: p[{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}] ({label})"
        )

    def _close_gripper(self):
        msg = Float32()
        msg.data = self.gripper_close
        self.pub_gripper.publish(msg)
        self.get_logger().info(f"  🤏 Gripper command: CLOSE ({self.gripper_close})")

    def _open_gripper(self):
        msg = Float32()
        msg.data = self.gripper_open
        self.pub_gripper.publish(msg)
        self.get_logger().info(f"  🖐️  Gripper command: OPEN ({self.gripper_open})")

    @staticmethod
    def _quat_to_rotvec(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm < 1e-9:
            return 0.0, 0.0, 0.0
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
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