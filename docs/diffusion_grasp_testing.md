# Diffusion Grasp Testing

## What Was Added

- `segmented_object_pcd_node`:
  - input: `depth + camera_info + mask`
  - output: segmented object `PointCloud2` in `camera_link`
- `diffusion_grasp_node`:
  - input: segmented object `PointCloud2` in 3D camera coordinates
  - output: `grasp_pose`, `grasp_pose_base`, `grasp_pose_gripper`, `grasp_candidates`, `best_score`

## Launch

From the ROS workspace:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch grasp_inference_pkg diffusion_grasp_from_depth.launch.py
```

## Bag-Based Testing

1. Make sure the bag contains:
   - depth image
   - camera info
   - binary object mask
   - tf/tf_static

2. Activate the GraspGen inference environment:

```bash
conda activate graspgen-infer
```

3. Source the ROS workspace in the same shell if your ROS 2 installation is available there.

4. Start the launch file above.

5. Play the bag:

```bash
ros2 bag play <bag_path> --clock
```

6. Inspect outputs:

- `/diffusion_grasp_node/grasp_pose`
- `/diffusion_grasp_node/grasp_pose_base`
- `/diffusion_grasp_node/grasp_pose_gripper`
- `/diffusion_grasp_node/grasp_candidates`
- `/diffusion_grasp_node/best_score`

## Notes

- `mask_topic` in the launch file should be changed to your real mask topic from the bag.
- This setup builds the segmented object cloud from depth, camera intrinsics, and mask.
- Before inference, the node recenters the point cloud around its mean the same way as the `MetaIsaacGrasp` remote pipeline, then restores predicted grasps back to the input cloud frame.
- In the default launch, inference runs on the `camera_link` cloud, `grasp_pose` stays in the input point cloud frame, and the node also republishes the best grasp in `base`.
- `grasp_pose_gripper` is computed with a full 6DoF offset transform, not only XYZ translation.
