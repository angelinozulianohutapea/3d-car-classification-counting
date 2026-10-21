"""Bandingkan panjang terukur deteksi fine-tune dengan deteksi model asli di lokasi yang sama."""
import pickle

import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

STRIDE, TOL_PX, MIN_SCORE = 32, 48.0, 0.3


def load(p):
    with open(p, "rb") as f:  # hanya muat file yang Anda buat sendiri
        return pickle.load(f)["frames"]


base, ft = load("det_low.pkl"), load("det_ft.pkl")
cam = to_world_from_camera[:3, 3]


def meters(b):
    g = get_position_ground(b["x"], b["y"], K, to_world_from_camera, 720)
    return g, float(np.linalg.norm(g - cam))


same, diff, new = [], [], []
for fb, ff in zip(base, ft):
    for b in ff:
        if b["score"] < MIN_SCORE:
            continue
        g, d = meters(b)
        if abs(g[0]) > 11:
            continue
        l_ft = b["l"] * d
        best, bd = None, TOL_PX
        for c in fb:
            dist = np.hypot(b["x"] - c["x"], b["y"] - c["y"])
            if dist <= bd:
                best, bd = c, dist
        if best is None:
            new.append(l_ft)
            continue
        l_base = best["l"] * meters(best)[1]
        same_cell = (int(b["x"] // STRIDE), int(b["y"] // STRIDE)) == (int(best["x"] // STRIDE), int(best["y"] // STRIDE))
        (same if same_cell else diff).append((l_ft, l_base))


def report(name, pairs):
    if not pairs:
        print(f"{name}: tidak ada")
        return
    a = np.array(pairs)
    print(f"{name}: n={len(a)} | panjang fine-tune median {np.median(a[:, 0]):.2f} m | "
          f"model asli median {np.median(a[:, 1]):.2f} m | rasio median {np.median(a[:, 0] / a[:, 1]):.2f}")


print("Deteksi fine-tune (skor >= 0.3, dalam jalur) dibandingkan dengan deteksi model asli terdekat:")
report("Sel sama   ", same)
report("Sel berbeda", diff)
if new:
    print(f"Tanpa pasangan di model asli: n={len(new)} | panjang median {np.median(new):.2f} m")
