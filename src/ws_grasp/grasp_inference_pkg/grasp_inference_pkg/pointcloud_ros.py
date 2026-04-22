from __future__ import annotations

import numpy as np

from sensor_msgs.msg import PointCloud2, PointField


R_OPTICAL_TO_LINK = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
T_OPTICAL_TO_LINK = np.zeros((3,), dtype=np.float32)


def normalize_frame_id(frame_id: str) -> str:
    frame_id = (frame_id or "").strip()
    while frame_id.startswith("/"):
        frame_id = frame_id[1:]
    return frame_id


def transform_to_matrix(transform_msg) -> np.ndarray:
    qx = transform_msg.transform.rotation.x
    qy = transform_msg.transform.rotation.y
    qz = transform_msg.transform.rotation.z
    qw = transform_msg.transform.rotation.w
    tx = transform_msg.transform.translation.x
    ty = transform_msg.transform.translation.y
    tz = transform_msg.transform.translation.z

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotated = points @ transform[:3, :3].T
    return rotated + transform[:3, 3][None, :]


def read_xyz_points(msg: PointCloud2) -> np.ndarray:
    field_map = {field.name: field for field in msg.fields}
    for name in ("x", "y", "z"):
        if name not in field_map or field_map[name].datatype != PointField.FLOAT32:
            raise ValueError(f"PointCloud2 field '{name}' must exist and be FLOAT32")

    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [np.float32, np.float32, np.float32],
            "offsets": [
                int(field_map["x"].offset),
                int(field_map["y"].offset),
                int(field_map["z"].offset),
            ],
            "itemsize": int(msg.point_step),
        }
    )
    count = int(msg.width) * int(max(msg.height, 1))
    arr = np.frombuffer(msg.data, dtype=dtype, count=count)
    points = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1)
    return points.astype(np.float32, copy=False)


def xyz_to_pointcloud2(points: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected Nx3 points, got shape {points.shape}")

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = int(points.shape[0])
    msg.is_dense = False
    msg.is_bigendian = False
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.data = points.astype(np.float32, copy=False).tobytes()
    return msg
