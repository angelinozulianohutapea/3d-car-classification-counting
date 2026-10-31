"""
Bandingkan konfigurasi detektor pada satu video.
Pakai: python experiments/test_detector_tuning.py --video data/video.mp4 --model centernet-3d-bbox.pth
"""
import argparse
import itertools
import time

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector import Detector

W, H = 1280, 720

CONFIGS = [
    # (nama, conf, scales, rgb)
    ("baseline conf0.30", 0.30, (1.0,), False),
    ("conf0.20", 0.20, (1.0,), False),
    ("conf0.15", 0.15, (1.0,), False),
    ("conf0.10", 0.10, (1.0,), False),
    ("conf0.15 + scale1.5", 0.15, (1.0, 1.5), False),
    ("conf0.15 + rgb", 0.15, (1.0,), True),
]


def read_frames(path, n, step):
    cap = cv2.VideoCapture(path)
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.resize(f, (W, H)))
        i += 1
    cap.release()
    return frames


def draw(frame, boxes, label):
    out = frame.copy()
    for b in boxes:
        x, y = int(b["x"]), int(b["y"])
        cv2.circle(out, (x, y), 6, (0, 255, 0), -1)
        cv2.putText(out, f'{b["score"]:.2f}', (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(out, f"{label} | n={len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--step", type=int, default=2, help="ambil tiap N frame")
    ap.add_argument("--save", default="experiments/out", help="folder hasil")
    args = ap.parse_args()

    import os
    os.makedirs(args.save, exist_ok=True)
    frames = read_frames(args.video, args.frames, args.step)
    print(f"{len(frames)} frame dimuat\n")

    print(f"{'config':26s} {'rata2 det':>10s} {'frame 0 det':>12s} {'FPS':>6s}")
    results = {}
    for name, conf, scales, rgb in CONFIGS:
        det = Detector(args.model, conf=conf, scales=scales, rgb=rgb)
        counts, outs = [], []
        t0 = time.time()
        for f in frames:
            boxes = det.detect(f)
            counts.append(len(boxes))
            outs.append(draw(f, boxes, name))
        fps = len(frames) / (time.time() - t0)
        results[name] = outs
        print(f"{name:26s} {np.mean(counts):10.2f} {sum(c == 0 for c in counts):12d} {fps:6.1f}")

        vw = cv2.VideoWriter(
            f"{args.save}/{name.replace(' ', '_').replace('+', '')}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H),
        )
        for o in outs:
            vw.write(o)
        vw.release()

    # gambar perbandingan: baseline vs konfigurasi terbaik pada 5 frame acak
    names = [c[0] for c in CONFIGS]
    idx = np.linspace(0, len(frames) - 1, 5, dtype=int)
    for i in idx:
        row = np.hstack([cv2.resize(results[n][i], (640, 360)) for n in (names[0], names[2], names[4])])
        cv2.imwrite(f"{args.save}/compare_{i:04d}.jpg", row)
    print(f"\nHasil tersimpan di {args.save}/")


if __name__ == "__main__":
    main()