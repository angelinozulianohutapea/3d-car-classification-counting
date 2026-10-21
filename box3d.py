"""
Kotak 3D berorientasi untuk tampilan: dimensi dan arah (yaw). Hanya geometri,
tidak memengaruhi tracking, klasifikasi, maupun counting.

Konvensi: x = lateral, y = tinggi (0 = jalan), z = kedalaman.
Seluruh pencampuran ukuran dan penghalusan yaw adalah [heuristik proyek].
"""

import math
from collections import deque

import numpy as np

import config as cfg


def blended_dims(cls, dim_buf):
    """(w, h, l) = campuran template kelas dan median terukur, di-clip per kelas."""
    tpl, lim = cfg.TEMPLATES[cls], cfg.LIMITS[cls]
    if len(dim_buf) == 0:
        return tpl["w"], tpl["h"], tpl["l"]

    med = np.median(np.array(dim_buf), axis=0)
    a = cfg.MEASURED_WEIGHT * min(1.0, len(dim_buf) / cfg.DIM_WARMUP)
    out = []
    for key, measured in zip(("w", "h", "l"), med):
        lo, hi = lim[key]
        out.append(float(np.clip((1 - a) * tpl[key] + a * measured, lo, hi)))
    return tuple(out)


class YawEstimator:
    """Arah kendaraan dari lintasan posisi, dihaluskan dengan EMA."""

    def __init__(self):
        self.pos_buf = deque(maxlen=cfg.YAW_WINDOW)
        self.yaw = 0.0
        # EMA pada sudut ganda (mobil simetris 180 derajat, tanpa masalah wrap-around)
        self._vec = np.array([1.0, 0.0])

    def update(self, x, z):
        self.pos_buf.append((x, z))
        if len(self.pos_buf) < 5:
            return
        x0, z0 = self.pos_buf[0]
        x1, z1 = self.pos_buf[-1]
        dx, dz = x1 - x0, z1 - z0
        if math.hypot(dx, dz) < cfg.YAW_MIN_DISPLACEMENT_M:
            return  # hampir diam, pertahankan yaw sebelumnya

        theta = math.atan2(dx, dz)  # 0 = lurus searah sumbu z
        if theta > math.pi / 2:  # maju atau mundur sama saja
            theta -= math.pi
        elif theta < -math.pi / 2:
            theta += math.pi

        meas = np.array([math.cos(2 * theta), math.sin(2 * theta)])
        self._vec = (1 - cfg.YAW_ALPHA) * self._vec + cfg.YAW_ALPHA * meas
        yaw = 0.5 * math.atan2(self._vec[1], self._vec[0])
        limit = math.radians(cfg.YAW_MAX_DEG)
        self.yaw = float(np.clip(yaw, -limit, limit))


def box_corners(cx, cz, dims, yaw):
    """8 sudut kotak di dunia. Indeks 0-3 = alas, 4-7 = atap."""
    w, h, l = dims
    f = np.array([math.sin(yaw), math.cos(yaw)])   # arah panjang
    r = np.array([math.cos(yaw), -math.sin(yaw)])  # arah lebar
    corners = []
    for height in (0.0, h):
        for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            p = np.array([cx, cz]) + sl * (l / 2) * f + sw * (w / 2) * r
            corners.append(np.array([p[0], height, p[1]]))
    return corners
