"""
Lihat skor deteksi per frame dari cache (mis. det_low.pkl) pada area tertentu.
Tujuan: membandingkan skor mobil nyata yang hilang-timbul dengan skor deteksi palsu.

Pakai:
  python analisis_cache.py det_low.pkl
Ubah kotak piksel / rentang frame di bagian KONFIGURASI sesuai kebutuhan.
"""
import pickle
import sys

import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

# --- KONFIGURASI -------------------------------------------------------
# A. SUV lajur kanan (perkiraan dari tangkapan layar frame 300): kotak piksel
SUV_FRAMES = range(290, 331)
SUV_BOX = (700, 900, 420, 720)  # koordinat video

# B. Dua "truk" palsu dari log: track 3 (frame 43-72) dan track 7 (frame 119-162),
#    posisi dunia y ~ 23 m, x 1-3 m (dari tabel "tidak dihitung")
FALSE_FRAMES = list(range(230, 331))
FALSE_Y_RANGE = (8.0, 45.0)
FALSE_X_RANGE = (2.0, 5.5)
# -----------------------------------------------------------------------


def ground(box):
    p = get_position_ground(box["x"], box["y"], K, to_world_from_camera, 720)
    return float(p[0]), float(p[2])


def main(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)  # hanya muat file yang Anda buat sendiri
    frames = cache["frames"]
    print(f"Cache: {path} | nms={cache['nms']} | skor minimum={cache['min_conf']} | {len(frames)} frame\n")

    print("A. SUV lajur kanan: deteksi di dalam kotak piksel (skor tertinggi per frame)")
    x0, x1, y0, y1 = SUV_BOX
    for i in SUV_FRAMES:
        hits = [b for b in frames[i] if x0 <= b["x"] <= x1 and y0 <= b["y"] <= y1]
        if hits:
            b = max(hits, key=lambda d: d["score"])
            print(f"  frame {i}: skor {b['score']:.3f}  piksel ({b['x']:.0f},{b['y']:.0f})")
        else:
            print(f"  frame {i}: tidak ada deteksi")

    print("\nB. Kandidat deteksi palsu (posisi dunia dalam rentang), skor tertinggi per frame")
    scores = []
    for i in FALSE_FRAMES:
        cands = []
        for b in frames[i]:
            x_m, y_m = ground(b)
            if FALSE_Y_RANGE[0] <= y_m <= FALSE_Y_RANGE[1] and FALSE_X_RANGE[0] <= x_m <= FALSE_X_RANGE[1]:
                cands.append((b, x_m, y_m))
        if cands:
            b, x_m, y_m = max(cands, key=lambda c: c[0]["score"])
            scores.append(b["score"])
            print(f"  frame {i}: skor {b['score']:.3f}  piksel ({b['x']:.0f},{b['y']:.0f})  dunia x={x_m:.1f} y={y_m:.1f}")
    if scores:
        print(f"\n  Ringkasan B: n={len(scores)}, skor min/median/maks = "
              f"{min(scores):.3f} / {np.median(scores):.3f} / {max(scores):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "det_low.pkl")
