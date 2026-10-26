"""
Metrik deteksi titik pusat terhadap label titik (labels.json).

Pencocokan: tiap deteksi (urut baris-per-baris) dicocokkan dengan label terdekat
yang belum terpakai dalam jarak <= TOL_PX piksel; hanya deteksi di dalam jalur
(LANE_X_MIN <= x <= LANE_X_MAX di config.py) yang dihitung. Karena label berupa titik,
IoU dan mAP tidak berlaku. Benar = TP, palsu = FP, terlewat = FN.

Precision, recall, F1: definisi standar, lihat D. M. W. Powers, "Evaluation:
From Precision, Recall and F-Measure to ROC, Informedness, Markedness and
Correlation", Journal of Machine Learning Technologies 2(1), 2011.
Toleransi 48 piksel (~1,5 sel) [heuristik proyek].
"""

import math

import config as cfg
from geometry import get_position_ground

TOL_PX = 48


def in_lane(x, y):
    return cfg.LANE_X_MIN <= get_position_ground(x, y)[0] <= cfg.LANE_X_MAX


def score_frame(dets, gts, tol=TOL_PX):
    """
    dets: [(x, y, skor)] sudah difilter jalur; gts: [[x, y]].
    Mengembalikan (cocok, palsu, indeks label terpakai).
    """
    used, matched, false_pos = set(), [], []
    for x, y, s in dets:
        best, best_d = None, tol
        for k, (gx, gy) in enumerate(gts):
            d = math.hypot(x - gx, y - gy)
            if k not in used and d <= best_d:
                best, best_d = k, d
        if best is None:
            false_pos.append((x, y, s))
        else:
            used.add(best)
            matched.append((x, y, s))
    return matched, false_pos, used


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def summary(tp, fp, fn):
    p, r, f1 = prf(tp, fp, fn)
    return f"benar {tp}, palsu {fp}, terlewat {fn} | precision {p:.3f}, recall {r:.3f}, F1 {f1:.3f}"
