"""
Pasca-proses deteksi: proyeksikan titik kontak ban ke bidang jalan, ubah
dimensi relatif model menjadi meter, lalu buang deteksi yang tidak masuk akal.

Keluaran detektor: x, y (piksel), w, h, l (rasio terhadap jarak). Meter =
rasio * depth, dengan depth = jarak titik kontak dari kamera [kode asal,
proyek antoinekeller/3d_bbox_highway]. Ambang filter: lihat config.py [heuristik].
"""

import numpy as np

import config as cfg
from geometry import CAMERA_POS, get_position_ground


def reject_reason(box, ground, depth):
    """Alasan deteksi dibuang, atau None bila lolos."""
    if abs(ground[0]) > cfg.LANE_HALF_WIDTH_M:
        return "luar jalur"
    if box["w"] * depth < cfg.MIN_WIDTH_M:
        return "w kecil"
    h_lo, h_hi = cfg.HEIGHT_RANGE_M
    if box["h"] * depth < h_lo or box["h"] * depth > h_hi:
        return "tinggi"
    if box["l"] * depth < cfg.MIN_LENGTH_M:
        return "l pendek"
    if box["h"] > box["l"]:
        return "h>l"
    return None


def to_world_detections(boxes, conf_split=None):
    """
    boxes      : keluaran Detector (atau cache).
    conf_split : ambang --conf bila asosiasi dua tahap aktif (hanya untuk
                 menandai deteksi skor rendah di mode --debug).
    Mengembalikan (deteksi dunia, tanda debug, hitungan ditolak per alasan).
    Tanda debug: (x_px, y_px, alasan atau None, skor_rendah).
    """
    world, marks, rejected = [], [], {}
    for box in boxes:
        ground = get_position_ground(box["x"], box["y"])
        depth = float(np.linalg.norm(ground - CAMERA_POS))
        reason = reject_reason(box, ground, depth)
        low = conf_split is not None and box["score"] < conf_split
        marks.append((box["x"], box["y"], reason, low))
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        world.append(
            {
                "x": float(ground[0]),
                "y": float(ground[2]),
                "w": min(box["w"] * depth, cfg.MAX_WIDTH_M),
                "h": box["h"] * depth,
                "l": box["l"] * depth,
                "score": box["score"],
            }
        )
    return world, marks, rejected
