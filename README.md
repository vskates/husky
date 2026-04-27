# HuskyRLGrasp

ROS 2 workspace for grasp inference on a robot.

## Current Diffusion Pipeline

This repository now includes a depth-based pipeline for testing a diffusion grasp model from `~/graspgen`.

Pipeline:

1. `depth + camera_info + mask`
2. `segmented_object_pcd_node`
3. segmented object `PointCloud2`
4. `diffusion_grasp_node`
5. grasp pose outputs

## Added Nodes

- `segmented_object_pcd_node`
  - input: depth image, camera info, binary object mask
  - output: segmented object `PointCloud2` in `camera_link`
- `diffusion_grasp_node`
  - input: segmented object `PointCloud2` in 3D camera coordinates
  - output:
    - `/diffusion_grasp_node/grasp_pose`
    - `/diffusion_grasp_node/grasp_pose_base`
    - `/diffusion_grasp_node/grasp_pose_gripper`
    - `/diffusion_grasp_node/grasp_candidates`
    - `/diffusion_grasp_node/best_score`

## Test Data Required

For bag-based testing you need:

- depth image
- camera info
- binary object mask
- `/tf`
- `/tf_static`

## Launch

Build the package:

```bash
colcon build --base-paths src/ws_grasp --packages-select grasp_inference_pkg --symlink-install
source install/setup.bash
```

Run the diffusion pipeline:

```bash
ros2 launch grasp_inference_pkg diffusion_grasp_from_depth.launch.py
```

## Bag Testing

1. Activate the GraspGen inference environment:

```bash
conda activate graspgen-infer
```

2. Update `mask_topic` in `src/ws_grasp/grasp_inference_pkg/launch/diffusion_grasp_from_depth.launch.py` to match your bag.

3. Start the launch file.

4. Play the bag:

```bash
ros2 bag play <bag_path> --clock
```

## Notes

- The segmented point cloud is built from depth, intrinsics, and mask.
- By default the segmented point cloud is published in `camera_link`, while `grasp_pose_base` and `grasp_pose_gripper` are published in `base`.
- `grasp_pose_gripper` is computed with a full 6DoF transform offset.
- The diffusion backend uses `~/graspgen` and prefers the `graspgen-infer` conda environment.

See also:

- `docs/diffusion_grasp_testing.md`
- `docs/realsense-calibration-guide.md`
