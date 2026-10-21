"""Pembuat aliran deteksi sintetis (tanpa video dan tanpa model) untuk menguji tracker dan counter."""

import numpy as np

DT = 1.0 / 24.0

# (kelas yang diharapkan, panjang sebenarnya m)
SPECS = {"car": (1.85, 1.5, 4.5), "van": (2.0, 2.0, 5.8), "truck": (2.5, 3.2, 9.5)}


def make_stream(n_frames=320, seed=0, n_vehicles=10, dropout=0.12, fp_rate=0.05):
    """
    Kendaraan mendekati kamera (y mengecil) dari 36 m ke 4 m dengan kecepatan 10-25 m/s.
    Deteksi diberi noise, kadang hilang (dropout), plus deteksi palsu acak.
    Mengembalikan (daftar deteksi per frame, daftar kendaraan sebenarnya).
    """
    rng = np.random.default_rng(seed)
    kinds = ["car", "car", "van", "truck", "car", "van", "car", "truck", "car", "car"]
    lanes = [-6.5, -3.0, 0.5, 4.0, 7.0]  # lajur bergantian: tidak ada dua kendaraan saling menyalip di lajur yang sama
    vehicles = []
    for k in range(n_vehicles):
        kind = kinds[k % len(kinds)]
        vehicles.append(
            {
                "kind": kind,
                "spawn": int(k * 28 + rng.integers(0, 8)),
                "x": float(lanes[k % len(lanes)] + rng.normal(0, 0.2)),
                "y0": 36.0,
                "vy": -float(rng.uniform(10, 25)),
                "dims": SPECS[kind],
            }
        )
    frames = []
    for f in range(n_frames):
        dets = []
        for v in vehicles:
            t = (f - v["spawn"]) * DT
            if t < 0:
                continue
            y = v["y0"] + v["vy"] * t
            if y < 3.0:
                continue
            if rng.random() < dropout:
                continue
            w, h, l = v["dims"]
            dets.append(
                {
                    "x": float(v["x"] + rng.normal(0, 0.15)),
                    "y": float(y + rng.normal(0, 0.3)),
                    "w": float(w + rng.normal(0, 0.1)),
                    "h": float(h + rng.normal(0, 0.1)),
                    "l": float(l + rng.normal(0, 0.2)),
                    "score": float(rng.uniform(0.3, 0.95)),
                }
            )
        if rng.random() < fp_rate:
            dets.append(
                {
                    "x": float(rng.uniform(-8, 8)),
                    "y": float(rng.uniform(8, 35)),
                    "w": 1.8, "h": 1.5, "l": 4.4,
                    "score": float(rng.uniform(0.3, 0.8)),
                }
            )
        rng.shuffle(dets)
        frames.append(dets)
    return frames, vehicles
