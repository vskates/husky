from setuptools import setup

package_name = "grasp_inference_pkg"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ('share/grasp_inference_pkg/models', [
            'models/model_5700.pth',
            'models/yolo_finetuned.pt',   
        ]),
        ("share/" + package_name + "/launch", ["launch/grasp_inference.launch.py"]),
    ],
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
            "camera_calibration = grasp_inference_pkg.camera_calibration:main",
            "gripper_exec = grasp_inference_pkg.gripper_exec:main",
        ],
    },
)
