"""
Geometri kamera: proyeksi 3D -> 2D dan proyeksi balik piksel ke bidang jalan.

Konvensi dunia: x = kanan, y = atas (0 = permukaan jalan), z = maju.
Pada Track, "y" milik Kalman Filter adalah z dunia.

Dasar: model kamera lubang jarum dan perpotongan sinar dengan bidang
y = 0 (Hartley & Zisserman, Multiple View Geometry, 2004, bab 6).
"""

import numpy as np

import config as cfg


def pitch_matrix(angle):
    """Rotasi sudut `angle` (radian) di sumbu X."""
    m = np.zeros((3, 3))
    m[0, 0] = 1
    m[1, 1] = np.cos(angle)
    m[1, 2] = np.sin(angle)
    m[2, 1] = -np.sin(angle)
    m[2, 2] = np.cos(angle)
    return m


def build_camera(pitch_deg, height_m, focal_px, img_w, img_h):
    """Kembalikan (K, to_world_from_camera). Titik pusat citra = tengah frame."""
    to_world = np.identity(4)
    to_world[:3, :3] = pitch_matrix(-pitch_deg * np.pi / 180)
    to_world[:3, 3] = np.array([0.0, height_m, 0.0])
    K = np.array(
        [[focal_px, 0.0, img_w / 2], [0.0, focal_px, img_h / 2], [0.0, 0.0, 1.0]]
    )
    return K, to_world


K, TO_WORLD_FROM_CAMERA = build_camera(
    cfg.PITCH_DEG, cfg.CAM_HEIGHT_M, cfg.FOCAL_PX, cfg.IMG_W, cfg.IMG_H
)
TO_CAMERA_FROM_WORLD = np.linalg.inv(TO_WORLD_FROM_CAMERA)
CAMERA_POS = TO_WORLD_FROM_CAMERA[:3, 3].copy()


def project(point_in_world, to_camera_from_world=TO_CAMERA_FROM_WORLD, k=K):
    """Titik dunia (3,) -> piksel (u, v) dengan v dihitung dari BAWAH citra."""
    p = np.asarray(point_in_world, dtype=float).reshape(3, 1)
    pc = to_camera_from_world[:3, :3].dot(p) + to_camera_from_world[:3, 3].reshape(3, 1)
    pc /= abs(pc[2])
    return k.dot(pc).flatten()[:2]


def world_to_image(point_in_world, img_h=cfg.IMG_H):
    """Titik dunia -> (u, v) piksel OpenCV (v dari ATAS citra), bertipe int."""
    pix = project(point_in_world)
    return int(pix[0]), int(img_h - pix[1])


def get_position_ground(x, y, img_h=cfg.IMG_H):
    """Piksel (x, y) OpenCV -> posisi dunia pada bidang jalan (y dunia = 0)."""
    pc = np.linalg.inv(K).dot(np.array([x, img_h - y, 1.0]).reshape(3, 1))
    pw = TO_WORLD_FROM_CAMERA[:3, :3].dot(pc) + CAMERA_POS.reshape(3, 1)
    direction = pw - CAMERA_POS.reshape(3, 1)
    direction /= np.linalg.norm(direction)
    lam = -CAMERA_POS[1] / direction[1]
    return (CAMERA_POS.reshape(3, 1) + lam * direction).flatten()
