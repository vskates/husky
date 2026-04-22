from setuptools import setup
from glob import glob

package_name = "grasp_inference_pkg"

launch_files = glob("launch/*.launch.py")
model_files = glob("models/*")

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]

if model_files:
    data_files.append((f"share/{package_name}/models", model_files))

if launch_files:
    data_files.append((f"share/{package_name}/launch", launch_files))

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools","numpy","opencv-python"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@todo.todo",
    description="GraspNet inference node",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grasp_node = grasp_inference_pkg.grasp_node:main",
            "model_forward = grasp_inference_pkg.model_forward:main",
            "segmented_object_pcd_node = grasp_inference_pkg.segmented_object_pcd_node:main",
            "diffusion_grasp_node = grasp_inference_pkg.diffusion_grasp_node:main",
            "gripper_exec = grasp_inference_pkg.gripper_exec:main",
        ],
    },
)
