"""
Estimasi bentuk kendaraan yang fleksibel:
- kelas kendaraan (car / van / truck) dengan voting
- dimensi = campuran template kelas dan ukuran terukur (median), di-clip per kelas
- yaw dari lintasan posisi, dihaluskan
- menggambar kotak 3D berorientasi (oriented box)

Konvensi dunia (sama seperti tracker):
    x = lateral, y = tinggi (0 = jalan), z = kedalaman/arah maju.
Pada Object di tracker, "y" milik Kalman = z dunia.
"""

import math
from collections import deque

import cv2
import numpy as np

from utils import project

# ----------------------------------------------------------------------
# Template dan batas ukuran per kelas (meter)
# ----------------------------------------------------------------------
TEMPLATES = {
    "car": {"w": 1.85, "h": 1.50, "l": 4.50},
    "van": {"w": 2.00, "h": 2.00, "l": 5.60},
    "truck": {"w": 2.50, "h": 3.20, "l": 9.00},
}

LIMITS = {
    "car": {"w": (1.6, 2.1), "h": (1.2, 1.9), "l": (3.5, 5.3)},
    "van": {"w": (1.8, 2.3), "h": (1.7, 2.6), "l": (4.8, 7.0)},
    "truck": {"w": (2.2, 2.6), "h": (2.5, 4.2), "l": (6.5, 13.0)},
}

# Seberapa besar ukuran terukur dipercaya (0 = template saja, 1 = ukuran terukur saja)
MEASURED_WEIGHT = 0.5

# Pengaturan yaw
YAW_WINDOW = 10  # jumlah posisi terakhir yang dipakai
YAW_MIN_DISPLACEMENT = 1.0  # meter, di bawah ini yaw tidak diubah
YAW_ALPHA = 0.2  # EMA, makin kecil makin halus
YAW_MAX = math.radians(35)  # mobil di jalan raya jarang lebih dari ini


def classify_by_length(length):
    if length < 5.2:
        return "car"
    if length < 7.0:
        return "van"
    return "truck"


class VehicleShape:
    """Menyimpan kelas, dimensi, dan yaw satu kendaraan."""

    def __init__(self):
        self.votes = {"car": 0, "van": 0, "truck": 0}
        self.cls = "car"
        self.dim_buf = deque(maxlen=30)
        self.pos_buf = deque(maxlen=YAW_WINDOW)
        self.yaw = 0.0
        # yaw dihaluskan dalam bentuk sudut ganda (mobil simetris 180 derajat)
        self._vec = np.array([1.0, 0.0])

    # ------------------------------------------------------------------
    def update(self, x, z, w, h, l):
        """Panggil setiap kali objek berhasil di-match dengan deteksi."""
        # --- kelas (voting) ---
        self.votes[classify_by_length(l)] += 1
        self.cls = max(self.votes, key=self.votes.get)

        # --- dimensi terukur ---
        self.dim_buf.append((w, h, l))

        # --- yaw ---
        self.pos_buf.append((x, z))
        self._update_yaw()

    # ------------------------------------------------------------------
    def dims(self):
        """Dimensi (w, h, l) final: campuran template dan median terukur."""
        tpl = TEMPLATES[self.cls]
        lim = LIMITS[self.cls]

        if len(self.dim_buf) == 0:
            return tpl["w"], tpl["h"], tpl["l"]

        med = np.median(np.array(self.dim_buf), axis=0)
        # bobot ukuran terukur naik pelan-pelan seiring jumlah pengamatan
        a = MEASURED_WEIGHT * min(1.0, len(self.dim_buf) / 10.0)

        out = []
        for key, measured in zip(("w", "h", "l"), med):
            value = (1 - a) * tpl[key] + a * measured
            lo, hi = lim[key]
            out.append(float(np.clip(value, lo, hi)))

        return tuple(out)

    # ------------------------------------------------------------------
    def _update_yaw(self):
        if len(self.pos_buf) < 5:
            return

        x0, z0 = self.pos_buf[0]
        x1, z1 = self.pos_buf[-1]
        dx, dz = x1 - x0, z1 - z0

        if math.hypot(dx, dz) < YAW_MIN_DISPLACEMENT:
            return  # hampir diam, pertahankan yaw sebelumnya

        theta = math.atan2(dx, dz)  # 0 = lurus searah sumbu z

        # lipat ke rentang [-90, 90] derajat (arah maju atau mundur sama saja)
        if theta > math.pi / 2:
            theta -= math.pi
        elif theta < -math.pi / 2:
            theta += math.pi

        # EMA pada sudut ganda supaya tidak ada masalah wrap-around
        meas = np.array([math.cos(2 * theta), math.sin(2 * theta)])
        self._vec = (1 - YAW_ALPHA) * self._vec + YAW_ALPHA * meas

        yaw = 0.5 * math.atan2(self._vec[1], self._vec[0])
        self.yaw = float(np.clip(yaw, -YAW_MAX, YAW_MAX))


# ----------------------------------------------------------------------
# Menggambar kotak 3D berorientasi
# ----------------------------------------------------------------------
def box_corners(cx, cz, dims, yaw):
    """8 sudut kotak di dunia. Indeks 0-3 = alas, 4-7 = atap."""
    w, h, l = dims
    f = np.array([math.sin(yaw), math.cos(yaw)])  # arah panjang
    r = np.array([math.cos(yaw), -math.sin(yaw)])  # arah lebar

    corners = []
    for height in (0.0, h):
        for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            p = np.array([cx, cz]) + sl * (l / 2) * f + sw * (w / 2) * r
            corners.append(np.array([p[0], height, p[1]]))
    return corners


def draw_oriented_box(
    frame, cx, cz, dims, yaw, to_camera_from_world, K, color, img_h=720, fill=True
):
    corners = box_corners(cx, cz, dims, yaw)

    pts = []
    for c in corners:
        pix = project(c, to_camera_from_world, K)
        pts.append((int(pix[0]), int(img_h - pix[1])))  # flip y sama seperti display()

    if fill:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(pts[0:4], dtype=np.int32)], color)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # alas
        (4, 5), (5, 6), (6, 7), (7, 4),  # atap
        (0, 4), (1, 5), (2, 6), (3, 7),  # tiang
    ]
    for a, b in edges:
        cv2.line(frame, pts[a], pts[b], color, 2)

    return frame