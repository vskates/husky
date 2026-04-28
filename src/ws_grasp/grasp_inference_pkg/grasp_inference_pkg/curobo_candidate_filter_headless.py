from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import yaml


@dataclass
class FilterConfig:
    robot_config_path: str
    robot_info_path: str
    gripper_mesh_path: str
    collision_threshold: float
    pregrasp_backoff: float
    pregrasp_halfspace_margin: float
    pregrasp_exit_eps: float
    curobo_enabled: bool
    curobo_batch_size: int
    curobo_max_iters: int
    curobo_num_ik_seeds: int
    curobo_num_graph_seeds: int
    curobo_num_trajopt_seeds: int
    curobo_timeout: float
    curobo_world_voxel_size: float
    curobo_world_buffer: float
    curobo_world_crop_margin: float
    curobo_max_world_cuboids: int
    curobo_position_threshold: float
    curobo_rotation_threshold: float
    start_joint_positions: list[float] | None
    current_ee_pose: list[float] | None
    force_cpu: bool
    collision_pose_batch: int = 16
    collision_scene_chunk: int = 4096
    collision_bbox_half: float = 0.1818
    gripper_surface_point_count: int = 1000

    @classmethod
    def from_json(cls, path: Path) -> "FilterConfig":
        payload = json.loads(path.read_text())
        return cls(**payload)


@dataclass
class SelectionResult:
    success: bool
    selected_grasp: np.ndarray
    selected_pregrasp: np.ndarray
    selected_score: float
    selected_index: int
    selected_was_flipped: bool
    filtered_grasps: np.ndarray
    filtered_scores: np.ndarray
    filtered_indices: np.ndarray
    reason: str


@dataclass
class IkBatchResult:
    success_mask: torch.Tensor
    position_error: torch.Tensor
    rotation_error: torch.Tensor
    solve_time_ms: float


@dataclass
class PlanResult:
    success: bool
    used: bool
    reason: str


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_robot_config_dict(path: Path) -> dict:
    robot_cfg = _load_yaml(path)
    kin = robot_cfg.get("robot_cfg", {}).get("kinematics", {})
    collision_spheres = kin.get("collision_spheres")
    if isinstance(collision_spheres, str) and not Path(collision_spheres).is_absolute():
        kin["collision_spheres"] = str((path.parent / collision_spheres).resolve())
    if kin.get("self_collision_buffer") is None:
        kin["self_collision_buffer"] = {}
    if kin.get("self_collision_ignore") is None:
        kin["self_collision_ignore"] = {}
    return robot_cfg


def _load_robot_info(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_float32_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-6)
    return points[valid].astype(np.float32, copy=False)


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T) + transform[:3, 3][None, :]


def _make_flip_rotation(dtype=np.float32) -> np.ndarray:
    rot = np.eye(4, dtype=dtype)
    rot[0, 0] = -1.0
    rot[1, 1] = -1.0
    return rot


def _matrix_stack_left_multiply(transform: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", transform, poses, dtype=np.float32).astype(np.float32, copy=False)


class HeadlessCuRoboPlanner:
    def __init__(self, cfg: FilterConfig) -> None:
        self.cfg = cfg
        self.robot_cfg_path = Path(cfg.robot_config_path).expanduser().resolve()
        self.robot_info_path = Path(cfg.robot_info_path).expanduser().resolve()
        self.robot_cfg_dict = _load_robot_config_dict(self.robot_cfg_path)
        self.robot_info = _load_robot_info(self.robot_info_path)
        cspace = self.robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]
        self.arm_joint_names = list(cspace["joint_names"])
        self.retract_config = np.asarray(cspace["retract_config"], dtype=np.float32)
        self.device = torch.device("cpu" if cfg.force_cpu or not torch.cuda.is_available() else "cuda:0")

        self._tensor_args = None
        self._motion_gen = None
        self._ik_solver = None
        self._plan_config = None
        self._Pose = None
        self._JointState = None
        self._WorldConfig = None
        self._Cuboid = None

        if cfg.curobo_enabled:
            self._ensure_backend()

    def _ensure_backend(self) -> None:
        if self._motion_gen is not None:
            return

        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

        with torch.enable_grad():
            tensor_args = TensorDeviceType(device=self.device)
            initial_world = WorldConfig(
                cuboid=[
                    Cuboid(
                        name="dummy_far_world_obstacle",
                        pose=[10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                        dims=[0.01, 0.01, 0.01],
                    )
                ],
                sphere=[],
                mesh=[],
                capsule=[],
                cylinder=[],
                blox=[],
                voxel=[],
            )
            motion_gen_cfg = MotionGenConfig.load_from_robot_config(
                self.robot_cfg_dict,
                initial_world,
                tensor_args,
                num_ik_seeds=self.cfg.curobo_num_ik_seeds,
                num_graph_seeds=self.cfg.curobo_num_graph_seeds,
                num_trajopt_seeds=self.cfg.curobo_num_trajopt_seeds,
                interpolation_dt=0.02,
                position_threshold=self.cfg.curobo_position_threshold,
                rotation_threshold=self.cfg.curobo_rotation_threshold,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                collision_cache={"obb": self.cfg.curobo_max_world_cuboids},
                use_cuda_graph=True,
                high_precision=True,
                evaluate_interpolated_trajectory=True,
                collision_activation_distance=max(self.cfg.curobo_world_buffer, 0.01),
                self_collision_check=False,
                self_collision_opt=False,
            )
            motion_gen = MotionGen(motion_gen_cfg)
            motion_gen.warmup(warmup_js_trajopt=False)

            ik_solver_cfg = IKSolverConfig.load_from_robot_config(
                self.robot_cfg_dict.get("robot_cfg", self.robot_cfg_dict),
                initial_world,
                tensor_args,
                num_seeds=self.cfg.curobo_num_ik_seeds,
                position_threshold=self.cfg.curobo_position_threshold,
                rotation_threshold=self.cfg.curobo_rotation_threshold,
                collision_checker_type=CollisionCheckerType.PRIMITIVE,
                collision_cache={"obb": self.cfg.curobo_max_world_cuboids},
                use_cuda_graph=True,
                self_collision_check=False,
                self_collision_opt=False,
                collision_activation_distance=max(self.cfg.curobo_world_buffer, 0.01),
                high_precision=True,
                store_debug=False,
            )
            ik_solver = IKSolver(ik_solver_cfg)
            plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                max_attempts=4,
                timeout=self.cfg.curobo_timeout,
                enable_graph_attempt=2,
                partial_ik_opt=False,
                enable_finetune_trajopt=True,
                check_start_validity=False,
            )

        self._tensor_args = tensor_args
        self._motion_gen = motion_gen
        self._ik_solver = ik_solver
        self._plan_config = plan_config
        self._Pose = Pose
        self._JointState = JointState
        self._WorldConfig = WorldConfig
        self._Cuboid = Cuboid

    def resolve_start_joint_positions(self) -> np.ndarray:
        if self.cfg.start_joint_positions is not None:
            start = np.asarray(self.cfg.start_joint_positions, dtype=np.float32).reshape(-1)
            if start.shape[0] != len(self.arm_joint_names):
                raise ValueError(
                    f"expected {len(self.arm_joint_names)} start joints, got {start.shape[0]}"
                )
            return start.astype(np.float32, copy=False)
        return self.retract_config.astype(np.float32, copy=False)

    def batch_goal_ik_feasible(self, target_poses: np.ndarray, start_joint_positions: np.ndarray) -> IkBatchResult:
        if not self.cfg.curobo_enabled:
            success = torch.ones((target_poses.shape[0],), dtype=torch.bool, device=self.device)
            zeros = torch.zeros((target_poses.shape[0],), dtype=torch.float32, device=self.device)
            return IkBatchResult(success, zeros, zeros, 0.0)

        self._ensure_backend()
        target_matrix = torch.as_tensor(
            target_poses,
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        ).view(-1, 4, 4)
        if target_matrix.shape[0] == 0:
            empty_bool = torch.zeros((0,), dtype=torch.bool, device=self.device)
            empty_float = torch.zeros((0,), dtype=torch.float32, device=self.device)
            return IkBatchResult(empty_bool, empty_float, empty_float, 0.0)

        problem_count = target_matrix.shape[0]
        batch_size = max(problem_count, int(self.cfg.curobo_batch_size))
        if batch_size > problem_count:
            pad = target_matrix[:1].expand(batch_size - problem_count, -1, -1)
            target_matrix = torch.cat((target_matrix, pad), dim=0)

        start_joint_pos = torch.as_tensor(
            start_joint_positions,
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        ).view(1, -1)
        retract_config = start_joint_pos.expand(batch_size, -1).clone()
        seed_config = start_joint_pos.view(1, 1, -1).expand(batch_size, 1, -1).clone()

        with torch.enable_grad():
            t0 = time.perf_counter()
            goal_pose = self._Pose.from_matrix(target_matrix)
            ik_result = self._ik_solver.solve_batch(
                goal_pose,
                retract_config=retract_config,
                seed_config=seed_config,
                return_seeds=1,
                num_seeds=self.cfg.curobo_num_ik_seeds,
                use_nn_seed=False,
                newton_iters=self.cfg.curobo_max_iters,
            )
            solve_time_ms = (time.perf_counter() - t0) * 1000.0

        success_mask = torch.as_tensor(ik_result.success, device=self.device).reshape(batch_size, -1)[:problem_count, 0]
        position_error = torch.as_tensor(ik_result.position_error, device=self.device).reshape(batch_size, -1)[:problem_count, 0]
        rotation_error = torch.as_tensor(ik_result.rotation_error, device=self.device).reshape(batch_size, -1)[:problem_count, 0]
        return IkBatchResult(success_mask, position_error, rotation_error, solve_time_ms)

    def plan_pregrasp_path(
        self,
        pregrasp_pose: np.ndarray,
        scene_points: np.ndarray,
        start_joint_positions: np.ndarray,
        current_ee_pose: np.ndarray | None,
    ) -> PlanResult:
        if not self.cfg.curobo_enabled:
            return PlanResult(success=True, used=False, reason="curobo disabled")

        try:
            self._ensure_backend()
            start_state = self._JointState.from_position(
                torch.as_tensor(
                    start_joint_positions,
                    device=self._tensor_args.device,
                    dtype=self._tensor_args.dtype,
                ).view(1, -1),
                joint_names=self.arm_joint_names,
            )
            anchor_positions = self._build_anchor_positions(current_ee_pose, [pregrasp_pose])
            world = self._build_world_from_scene_points(scene_points, anchor_positions)
            self._motion_gen.reset(reset_seed=False)
            self._motion_gen.update_world(world)

            target_matrix = torch.as_tensor(
                pregrasp_pose,
                device=self._tensor_args.device,
                dtype=self._tensor_args.dtype,
            ).view(1, 4, 4)
            goal_pose = self._Pose.from_matrix(target_matrix)
            result = self._motion_gen.plan_single(start_state, goal_pose, self._plan_config)
            success_tensor = getattr(result, "success", None)
            if success_tensor is None or not bool(success_tensor.reshape(-1)[0].item()):
                reason = getattr(getattr(result, "status", None), "value", "planning failed")
                return PlanResult(success=False, used=True, reason=str(reason))
            interpolated_plan = result.get_interpolated_plan()
            joint_path = getattr(interpolated_plan, "position", None) if interpolated_plan is not None else None
            if joint_path is None:
                return PlanResult(success=False, used=True, reason="interpolated joint path missing")
            return PlanResult(success=True, used=True, reason="planned current->pregrasp path")
        except Exception as exc:
            return PlanResult(success=False, used=True, reason=f"planner execution failed: {exc}")

    def _build_anchor_positions(self, current_ee_pose: np.ndarray | None, target_poses: list[np.ndarray]) -> np.ndarray:
        anchors: list[np.ndarray] = []
        if current_ee_pose is not None:
            current = np.asarray(current_ee_pose, dtype=np.float32)
            if current.shape == (4, 4):
                anchors.append(current[:3, 3].copy())
            elif current.shape[-1] >= 3:
                anchors.append(current[:3].copy())
        for pose in target_poses:
            pose_arr = np.asarray(pose, dtype=np.float32).reshape(4, 4)
            anchors.append(pose_arr[:3, 3].copy())
        if not anchors:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(anchors, axis=0).astype(np.float32, copy=False)

    def _build_world_from_scene_points(self, scene_points: np.ndarray, anchor_positions: np.ndarray):
        points = np.asarray(scene_points, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if points.size == 0:
            return self._empty_world()
        if float(np.max(np.abs(points))) > 10.0:
            points = points / 1000.0

        if anchor_positions.size > 0:
            crop_low = anchor_positions.min(axis=0) - self.cfg.curobo_world_crop_margin
            crop_high = anchor_positions.max(axis=0) + self.cfg.curobo_world_crop_margin
            in_crop = np.logical_and(points >= crop_low, points <= crop_high).all(axis=1)
            cropped = points[in_crop]
            if cropped.shape[0] > 0:
                points = cropped

        voxel = max(self.cfg.curobo_world_voxel_size, 0.02)
        voxel_index = np.floor(points / voxel).astype(np.int32)
        unique_voxels = np.unique(voxel_index, axis=0)
        if unique_voxels.shape[0] == 0:
            return self._empty_world()

        voxel_centers = (unique_voxels.astype(np.float32) + 0.5) * voxel
        if anchor_positions.size > 0 and voxel_centers.shape[0] > self.cfg.curobo_max_world_cuboids:
            distances = np.linalg.norm(
                voxel_centers[:, None, :] - anchor_positions[None, :, :],
                axis=-1,
            ).min(axis=1)
            order = np.argsort(distances)[: self.cfg.curobo_max_world_cuboids]
            voxel_centers = voxel_centers[order]
        elif voxel_centers.shape[0] > self.cfg.curobo_max_world_cuboids:
            voxel_centers = voxel_centers[: self.cfg.curobo_max_world_cuboids]

        dims = [voxel + self.cfg.curobo_world_buffer] * 3
        cuboids = [
            self._Cuboid(
                name=f"pc_{idx}",
                pose=[float(center[0]), float(center[1]), float(center[2]), 1.0, 0.0, 0.0, 0.0],
                dims=dims,
            )
            for idx, center in enumerate(voxel_centers)
        ]
        return self._WorldConfig(
            cuboid=cuboids,
            sphere=[],
            mesh=[],
            capsule=[],
            cylinder=[],
            blox=[],
            voxel=[],
        )

    def _empty_world(self):
        return self._WorldConfig(
            cuboid=[
                self._Cuboid(
                    name="dummy_far_world_obstacle",
                    pose=[10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.01, 0.01, 0.01],
                )
            ],
            sphere=[],
            mesh=[],
            capsule=[],
            cylinder=[],
            blox=[],
            voxel=[],
        )


class HeadlessGraspCandidateFilter:
    def __init__(self, cfg: FilterConfig) -> None:
        self.cfg = cfg
        self.planner = HeadlessCuRoboPlanner(cfg)
        self.device = self.planner.device if cfg.curobo_enabled else torch.device(
            "cpu" if cfg.force_cpu or not torch.cuda.is_available() else "cuda:0"
        )
        self.gripper_surface_points_h = self._load_gripper_surface_points(
            Path(cfg.gripper_mesh_path).expanduser().resolve(),
            cfg.gripper_surface_point_count,
        )

    def _load_gripper_surface_points(self, mesh_path: Path, count: int) -> torch.Tensor:
        mesh = trimesh.load(mesh_path)
        mesh.apply_transform(tra.rotation_matrix(-0.5 * np.pi, [0.0, 0.0, 1.0]))
        mesh.apply_transform(tra.rotation_matrix(-0.5 * np.pi, [0.0, 1.0, 0.0]))
        mesh.apply_transform(tra.rotation_matrix(-0.5 * np.pi, [0.0, 0.0, 1.0]))
        surface_points, _ = trimesh.sample.sample_surface(mesh, count)
        surface_points_t = torch.as_tensor(surface_points, device=self.device, dtype=torch.float32)
        ones = torch.ones((surface_points_t.shape[0], 1), device=self.device, dtype=torch.float32)
        return torch.cat((surface_points_t, ones), dim=1)

    def select(self, grasps: np.ndarray, scores: np.ndarray, object_points: np.ndarray, scene_points: np.ndarray) -> SelectionResult:
        grasps = np.asarray(grasps, dtype=np.float32).reshape(-1, 4, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        object_points = _as_float32_points(object_points)
        scene_points = _as_float32_points(scene_points)

        if grasps.shape[0] == 0 or scores.shape[0] == 0:
            return self._failure("no grasp candidates")

        score_order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
        grasps = grasps[score_order]
        scores = scores[score_order]
        raw_indices = score_order.astype(np.int64, copy=False)

        pregrasp_poses = self._compute_pregrasp_poses(grasps, object_points)
        collision_free_mask, _, _ = self._fast_grasp_collision_check(
            grasps,
            scene_points,
            pregrasp_poses,
        )
        collision_free_indices = np.flatnonzero(collision_free_mask)
        if collision_free_indices.size == 0:
            return self._failure("all candidates rejected by collision filters")

        ordered_indices, ordered_actions, ordered_pregrasps, ordered_flipped = self._prioritize_z_flip_variants(
            collision_free_indices,
            grasps,
            pregrasp_poses,
        )
        start_joint_positions = self.planner.resolve_start_joint_positions()
        current_ee_pose = None
        if self.cfg.current_ee_pose is not None:
            current_ee_pose = np.asarray(self.cfg.current_ee_pose, dtype=np.float32)

        for chunk_start in range(0, ordered_indices.shape[0], self.cfg.curobo_batch_size):
            chunk_end = min(chunk_start + self.cfg.curobo_batch_size, ordered_indices.shape[0])
            chunk_indices = ordered_indices[chunk_start:chunk_end]
            chunk_actions = ordered_actions[chunk_start:chunk_end]
            chunk_pregrasps = ordered_pregrasps[chunk_start:chunk_end]
            chunk_flipped = ordered_flipped[chunk_start:chunk_end]
            ik_result = self.planner.batch_goal_ik_feasible(chunk_pregrasps, start_joint_positions)

            success_indices = torch.where(ik_result.success_mask)[0].detach().cpu().numpy().tolist()
            for local_idx in success_indices:
                plan_result = self.planner.plan_pregrasp_path(
                    pregrasp_pose=chunk_pregrasps[local_idx],
                    scene_points=scene_points,
                    start_joint_positions=start_joint_positions,
                    current_ee_pose=current_ee_pose,
                )
                if not (plan_result.success or not plan_result.used):
                    continue
                raw_sorted_idx = int(chunk_indices[local_idx])
                return SelectionResult(
                    success=True,
                    selected_grasp=chunk_actions[local_idx].astype(np.float32, copy=False),
                    selected_pregrasp=chunk_pregrasps[local_idx].astype(np.float32, copy=False),
                    selected_score=float(scores[raw_sorted_idx]),
                    selected_index=int(raw_indices[raw_sorted_idx]),
                    selected_was_flipped=bool(chunk_flipped[local_idx]),
                    filtered_grasps=chunk_actions[local_idx : local_idx + 1].astype(np.float32, copy=False),
                    filtered_scores=np.asarray([scores[raw_sorted_idx]], dtype=np.float32),
                    filtered_indices=np.asarray([raw_indices[raw_sorted_idx]], dtype=np.int64),
                    reason="selected first collision-free IK-feasible cuRobo-plannable candidate",
                )

        return self._failure("no candidate passed cuRobo IK + path planning")

    def _failure(self, reason: str) -> SelectionResult:
        identity = np.eye(4, dtype=np.float32)
        return SelectionResult(
            success=False,
            selected_grasp=identity,
            selected_pregrasp=identity,
            selected_score=float("-inf"),
            selected_index=-1,
            selected_was_flipped=False,
            filtered_grasps=np.zeros((0, 4, 4), dtype=np.float32),
            filtered_scores=np.zeros((0,), dtype=np.float32),
            filtered_indices=np.zeros((0,), dtype=np.int64),
            reason=reason,
        )

    def _compute_pregrasp_poses(self, grasp_poses: np.ndarray, object_pc: np.ndarray) -> np.ndarray:
        return np.stack(
            [self._compute_pregrasp_pose_from_object_pc(grasp_pose, object_pc) for grasp_pose in grasp_poses],
            axis=0,
        ).astype(np.float32, copy=False)

    def _compute_pregrasp_pose_from_object_pc(self, grasp_pose: np.ndarray, object_pc: np.ndarray) -> np.ndarray:
        grasp_pose = np.asarray(grasp_pose, dtype=np.float32).reshape(4, 4)
        if object_pc.size == 0:
            return self._fallback_pregrasp_pose(grasp_pose)

        mins = object_pc.min(axis=0)
        maxs = object_pc.max(axis=0)
        x_low = mins[0] - self.cfg.pregrasp_halfspace_margin
        y_high = maxs[1] + self.cfg.pregrasp_halfspace_margin
        y_low = mins[1] - self.cfg.pregrasp_halfspace_margin
        z_high = maxs[2] + self.cfg.pregrasp_halfspace_margin

        grasp_pos = grasp_pose[:3, 3]
        approach_axis = grasp_pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if (
            grasp_pos[0] <= x_low
            or grasp_pos[1] >= y_high
            or grasp_pos[1] <= y_low
            or grasp_pos[2] >= z_high
        ):
            travel = 0.0
        else:
            candidates: list[float] = []
            if approach_axis[0] > 1e-8:
                t_x_low = (grasp_pos[0] - x_low) / approach_axis[0]
                if t_x_low >= 0.0:
                    candidates.append(float(t_x_low))
            if approach_axis[1] < -1e-8:
                t_y_high = (y_high - grasp_pos[1]) / (-approach_axis[1])
                if t_y_high >= 0.0:
                    candidates.append(float(t_y_high))
            if approach_axis[1] > 1e-8:
                t_y_low = (grasp_pos[1] - y_low) / approach_axis[1]
                if t_y_low >= 0.0:
                    candidates.append(float(t_y_low))
            if approach_axis[2] < -1e-8:
                t_z_high = (z_high - grasp_pos[2]) / (-approach_axis[2])
                if t_z_high >= 0.0:
                    candidates.append(float(t_z_high))
            if not candidates:
                return self._fallback_pregrasp_pose(grasp_pose)
            travel = min(candidates)

        pregrasp_pose = grasp_pose.copy()
        pregrasp_pose[:3, 3] = grasp_pos - (travel + self.cfg.pregrasp_exit_eps) * approach_axis
        return pregrasp_pose.astype(np.float32, copy=False)

    def _fallback_pregrasp_pose(self, grasp_pose: np.ndarray) -> np.ndarray:
        pregrasp_pose = grasp_pose.copy()
        approach_offset_world = grasp_pose[:3, :3] @ np.array(
            [0.0, 0.0, self.cfg.pregrasp_backoff],
            dtype=np.float32,
        )
        pregrasp_pose[:3, 3] = grasp_pose[:3, 3] - approach_offset_world
        return pregrasp_pose.astype(np.float32, copy=False)

    def _fast_grasp_collision_check(
        self,
        grasps: np.ndarray,
        scene_points: np.ndarray,
        pregrasp_poses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if grasps.shape[0] == 0:
            empty = np.zeros((0,), dtype=bool)
            return empty, empty, empty

        scene_pc = torch.as_tensor(scene_points, device=self.device, dtype=torch.float32)
        grasps_t = torch.as_tensor(grasps, device=self.device, dtype=torch.float32)
        pregrasp_t = torch.as_tensor(pregrasp_poses, device=self.device, dtype=torch.float32)

        approach_x_mask = grasps_t[:, 0, 2] >= 0.0
        pregrasp_stage_mask = approach_x_mask.clone()
        surviving_indices = torch.where(pregrasp_stage_mask)[0]
        if surviving_indices.numel() > 0:
            pregrasp_collision_free = self._optimized_collision_check(scene_pc, pregrasp_t[surviving_indices])
            pregrasp_stage_mask = torch.zeros_like(pregrasp_stage_mask)
            pregrasp_stage_mask[surviving_indices] = pregrasp_collision_free

        grasp_stage_mask = torch.zeros_like(pregrasp_stage_mask)
        surviving_indices = torch.where(pregrasp_stage_mask)[0]
        if surviving_indices.numel() > 0:
            grasp_collision_free = self._optimized_collision_check(scene_pc, grasps_t[surviving_indices])
            grasp_stage_mask[surviving_indices] = grasp_collision_free

        collision_mask = grasp_stage_mask.clone()
        return (
            collision_mask.detach().cpu().numpy(),
            grasp_stage_mask.detach().cpu().numpy(),
            pregrasp_stage_mask.detach().cpu().numpy(),
        )

    def _optimized_collision_check(self, scene_pc: torch.Tensor, grasp_poses: torch.Tensor) -> torch.Tensor:
        if grasp_poses.numel() == 0:
            return torch.ones((0,), dtype=torch.bool, device=self.device)

        collision_free_mask = torch.ones((grasp_poses.shape[0],), dtype=torch.bool, device=self.device)
        threshold = float(self.cfg.collision_threshold)
        bbox_half = float(self.cfg.collision_bbox_half)

        for batch_start in range(0, grasp_poses.shape[0], self.cfg.collision_pose_batch):
            batch_end = min(batch_start + self.cfg.collision_pose_batch, grasp_poses.shape[0])
            grasp_pose_batch = grasp_poses[batch_start:batch_end]
            centers = grasp_pose_batch[:, :3, 3]
            min_corner = centers - bbox_half
            max_corner = centers + bbox_half
            local_scene_masks = (
                (scene_pc.unsqueeze(0) >= min_corner.unsqueeze(1))
                & (scene_pc.unsqueeze(0) <= max_corner.unsqueeze(1))
            ).all(dim=-1)

            world_pts = torch.matmul(
                grasp_pose_batch,
                self.gripper_surface_points_h.T.unsqueeze(0).expand(grasp_pose_batch.shape[0], -1, -1),
            ).transpose(1, 2)[..., :3]

            for local_idx in range(grasp_pose_batch.shape[0]):
                pose_idx = batch_start + local_idx
                local_scene = scene_pc[local_scene_masks[local_idx]]
                if local_scene.shape[0] == 0:
                    continue
                min_distance = float("inf")
                query_pts = world_pts[local_idx].unsqueeze(0)
                for scene_start in range(0, local_scene.shape[0], self.cfg.collision_scene_chunk):
                    scene_chunk = local_scene[scene_start : scene_start + self.cfg.collision_scene_chunk].unsqueeze(0)
                    chunk_min_distance = torch.cdist(query_pts, scene_chunk).amin()
                    min_distance = min(min_distance, float(chunk_min_distance.item()))
                    if min_distance < threshold:
                        collision_free_mask[pose_idx] = False
                        break
        return collision_free_mask

    def _prioritize_z_flip_variants(
        self,
        collision_free_indices: np.ndarray,
        actions: np.ndarray,
        pregrasps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if collision_free_indices.size == 0:
            return (
                collision_free_indices.astype(np.int64, copy=False),
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=bool),
            )

        candidate_actions = actions[collision_free_indices]
        candidate_pregrasps = pregrasps[collision_free_indices]
        local_y_points_down = candidate_actions[:, 2, 1] < 0.0

        flip_rot = _make_flip_rotation(dtype=np.float32)
        flipped_actions = candidate_actions.copy()
        flipped_pregrasps = candidate_pregrasps.copy()
        flipped_actions[:, :3, :3] = candidate_actions[:, :3, :3] @ flip_rot[:3, :3]
        flipped_pregrasps[:, :3, :3] = candidate_pregrasps[:, :3, :3] @ flip_rot[:3, :3]

        preferred_actions = np.where(local_y_points_down[:, None, None], flipped_actions, candidate_actions)
        preferred_pregrasps = np.where(local_y_points_down[:, None, None], flipped_pregrasps, candidate_pregrasps)
        alternate_actions = np.where(local_y_points_down[:, None, None], candidate_actions, flipped_actions)
        alternate_pregrasps = np.where(local_y_points_down[:, None, None], candidate_pregrasps, flipped_pregrasps)

        ordered_indices = np.concatenate((collision_free_indices, collision_free_indices), axis=0).astype(np.int64, copy=False)
        ordered_actions = np.concatenate((preferred_actions, alternate_actions), axis=0).astype(np.float32, copy=False)
        ordered_pregrasps = np.concatenate((preferred_pregrasps, alternate_pregrasps), axis=0).astype(np.float32, copy=False)
        ordered_flipped = np.concatenate((local_y_points_down, ~local_y_points_down), axis=0)
        return ordered_indices, ordered_actions, ordered_pregrasps, ordered_flipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter diffusion grasp candidates with MetaIsaacGrasp-style logic")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--grasps", required=True, help="Path to input grasps .npy")
    parser.add_argument("--scores", required=True, help="Path to input scores .npy")
    parser.add_argument("--object-point-cloud", required=True, help="Path to segmented object point cloud .npy")
    parser.add_argument("--scene-point-cloud", required=True, help="Path to full scene point cloud .npy")
    parser.add_argument("--output", required=True, help="Path to output .npz")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = FilterConfig.from_json(Path(args.config))
    grasps = np.load(args.grasps)
    scores = np.load(args.scores)
    object_points = np.load(args.object_point_cloud)
    scene_points = np.load(args.scene_point_cloud)

    selector = HeadlessGraspCandidateFilter(cfg)
    result = selector.select(grasps, scores, object_points, scene_points)

    np.savez(
        args.output,
        success=np.asarray(result.success, dtype=np.bool_),
        selected_grasp=result.selected_grasp.astype(np.float32, copy=False),
        selected_pregrasp=result.selected_pregrasp.astype(np.float32, copy=False),
        selected_score=np.asarray(result.selected_score, dtype=np.float32),
        selected_index=np.asarray(result.selected_index, dtype=np.int64),
        selected_was_flipped=np.asarray(result.selected_was_flipped, dtype=np.bool_),
        filtered_grasps=result.filtered_grasps.astype(np.float32, copy=False),
        filtered_scores=result.filtered_scores.astype(np.float32, copy=False),
        filtered_indices=result.filtered_indices.astype(np.int64, copy=False),
        reason=np.asarray(result.reason),
    )
    print(
        f"filter_success={result.success} selected_index={result.selected_index} "
        f"selected_score={result.selected_score} reason={result.reason}"
    )


if __name__ == "__main__":
    main()
