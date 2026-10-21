"""Lembar gambar: satu frame per hitungan, dengan penanda posisi saat melewati garis."""
import csv
from argparse import ArgumentParser

import cv2
import numpy as np

from multi_object_tracking import K, to_world_from_camera
from utils import project

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("--log", default="counting_base03.csv")
ap.add_argument("--line-y", type=float, default=15.0)
ap.add_argument("--per", type=int, default=9)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.log, encoding="utf-8")))
cap = cv2.VideoCapture(args.video)
to_cam = np.linalg.inv(to_world_from_camera)

tiles = []
for r in rows:
    i = int(r["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        continue
    h = fr.shape[0]
    pix = project(np.array([float(r["x_m"]), 0.0, args.line_y]), to_cam, K)
    cv2.circle(fr, (int(pix[0]), int(h - pix[1])), 14, (255, 0, 255), 4)
    cv2.putText(fr, f"ID {r['id']} | frame {i} | prediksi: {r['kelas']}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    tiles.append(cv2.resize(fr, (640, 360)))

for s in range(0, len(tiles), args.per):
    chunk = tiles[s:s + args.per]
    while len(chunk) < args.per:
        chunk.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(chunk[k:k + 3]) for k in range(0, args.per, 3)]
    name = f"lembar_{s // args.per + 1}.png"
    cv2.imwrite(name, np.vstack(rows_img))
    print("Disimpan:", name)
