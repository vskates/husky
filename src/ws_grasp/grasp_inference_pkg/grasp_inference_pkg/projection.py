from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class HeightmapSpec:
    size: int = 224
    resolution: float = 0.002  # meters per pixel
    # bounds for the 2 plane axes in the POINT FRAME (camera_frame or base_link), meters:
    plane_min: np.ndarray = np.array([-0.2, -0.2], dtype=np.float32)  # [u_min, v_min]
    plane_max: np.ndarray = np.array([ 0.2,  0.2], dtype=np.float32)  # [u_max, v_max]

    # mapping: which coordinates of XYZ are height axis and plane axes
    # default assumes points are already in a frame where:
    #   height axis = X, plane axes = (Y,Z) like in твоём симе
    height_axis: int = 0
    plane_axes: tuple[int, int] = (1, 2)


def depth_to_xyz(depth_m: np.ndarray, K: CameraIntrinsics) -> np.ndarray:
    """
    принимает карту глубины и CameraIntrinsics, выдаёт 3 мерное представление каждого пикселя. по сути 
    делается для того что бы получить численное значение глубины.
    depth_m: (H,W) float32 depth in meters, in CAMERA OPTICAL frame convention.
    Returns xyz: (H,W,3) float32 in camera frame.
    """
    h, w = depth_m.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth_m
    x = (uu - K.cx) * z / K.fx
    y = (vv - K.cy) * z / K.fy
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz


def build_heightmaps(
    rgb_u8: np.ndarray,               # (H,W,3) uint8
    xyz: np.ndarray,                  # (H,W,3) float32
    spec: HeightmapSpec,
    mask_u8: np.ndarray | None = None # (H,W) uint8, 0 background, >0 object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    принимает на вход
        rgb_u8 (H,W,3) uint8
        xyz (H,W,3) float32 из ф-ии depth_to_xyz
        spec - сепцификации Heightmap
        mask_u8 из под Yolo -  (H,W) uint8, 0 background, >0 object
    Returns:
      color_hm: (S,S,3) uint8
      height_hm: (S,S) float32
      mask_hm: (S,S) uint8
    """
    S = spec.size
    color_hm = np.zeros((S, S, 3), dtype=np.uint8)
    height_hm = np.zeros((S, S), dtype=np.float32)
    mask_hm = np.zeros((S, S), dtype=np.uint8)

    pts = xyz.reshape(-1, 3)
    rgb = rgb_u8.reshape(-1, 3)

    # valid depth
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 0)
    if mask_u8 is not None:
        m = mask_u8.reshape(-1)
        valid = valid & (m > 0)
    else:
        m = np.zeros((pts.shape[0],), dtype=np.uint8)

    pts = pts[valid]
    rgb = rgb[valid]
    m = m[valid]

    if pts.shape[0] == 0:
        return color_hm, height_hm, mask_hm

    # plane coords
    u_axis, v_axis = spec.plane_axes
    uv = pts[:, [u_axis, v_axis]]

    # bounds filter
    inb = (
        (uv[:, 0] >= spec.plane_min[0]) & (uv[:, 0] < spec.plane_max[0]) &
        (uv[:, 1] >= spec.plane_min[1]) & (uv[:, 1] < spec.plane_max[1])
    )
    pts = pts[inb]
    rgb = rgb[inb]
    m = m[inb]
    uv = uv[inb]

    if pts.shape[0] == 0:
        return color_hm, height_hm, mask_hm

    # pixel indices
    pix = np.floor((uv - spec.plane_min[None, :]) / spec.resolution).astype(np.int32)
    # clamp into [0, S-1]
    pix[:, 0] = np.clip(pix[:, 0], 0, S - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, S - 1)

    # flip to match your sim convention (size-1 - pix)
    px = (S - 1) - pix[:, 0]
    py = (S - 1) - pix[:, 1]

    hvals = pts[:, spec.height_axis].astype(np.float32)

    # simplest fill (как у тебя): последнее значение перезаписывает
    color_hm[py, px] = rgb
    height_hm[py, px] = hvals
    if mask_u8 is not None:
        mask_hm[py, px] = m

    return color_hm, height_hm, mask_hm
