"""
Lembar gambar untuk memeriksa hasil counting: satu frame per kendaraan yang
dihitung (dari counting_log.csv), dengan lingkaran ungu di posisi saat melewati garis.

Pakai (dari folder proyek):
  python -m tools.lembar_hitungan VIDEO.mp4 --log counting_log.csv --line-y 15
"""

import csv
from argparse import ArgumentParser

import cv2
import numpy as np

import config as cfg
from geometry import world_to_image


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("video")
    ap.add_argument("--log", default=cfg.COUNT_LOG)
    ap.add_argument("--line-y", type=float, default=cfg.LINE_Y_DEFAULT_M)
    ap.add_argument("--per", type=int, default=9, help="frame per lembar (kelipatan 3)")
    args = ap.parse_args()

    with open(args.log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cap = cv2.VideoCapture(args.video)

    tiles = []
    for r in rows:
        i = int(r["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        u, v = world_to_image(np.array([float(r["x_m"]), 0.0, args.line_y]), frame.shape[0])
        cv2.circle(frame, (u, v), 14, (255, 0, 255), 4)
        cv2.putText(frame, f"ID {r['id']} | frame {i} | prediksi: {r['kelas']}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        tiles.append(cv2.resize(frame, (640, 360)))
    cap.release()
    if not tiles:
        raise SystemExit("Tidak ada frame yang bisa dibaca")

    for s in range(0, len(tiles), args.per):
        chunk = tiles[s:s + args.per]
        while len(chunk) < args.per:
            chunk.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(chunk[k:k + 3]) for k in range(0, args.per, 3)])
        name = f"lembar_{s // args.per + 1}.png"
        cv2.imwrite(name, sheet)
        print("Disimpan:", name)


if __name__ == "__main__":
    main()
