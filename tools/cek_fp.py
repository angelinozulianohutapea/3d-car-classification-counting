"""
Evaluasi deteksi terhadap label titik dan simpan gambar frame dengan deteksi
palsu terbanyak. Dekode memakai offset, sama seperti Detector (pred ke piksel).
Lingkaran merah = label (lingkaran besar = terlewat), hijau = deteksi benar,
kuning = deteksi palsu (dengan skor). Metrik: lihat tools/metrics.py.

Pakai (dari folder proyek):
  python -m tools.cek_fp VIDEO.mp4 MODEL.pth --conf 0.3 --from 330 --to 472 --out cek_ft
"""

import json
import os
from argparse import ArgumentParser

import cv2

from detector import Detector, decode_points
from tools.metrics import in_lane, score_frame, summary


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("video")
    ap.add_argument("model")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--from", dest="f0", type=int, default=330)
    ap.add_argument("--to", dest="f1", type=int, default=472)
    ap.add_argument("--top", type=int, default=8, help="jumlah gambar frame palsu terbanyak yang disimpan")
    ap.add_argument("--out", default="cek_fp")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        labels = {int(k): v["points"] for k, v in json.load(f).items() if v["done"]}

    detector = Detector(args.model, args.conf)
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)

    tp = fp = fn = 0
    per_frame = []
    for i in range(args.f0, args.f1 + 1):
        if i not in labels:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        hm, offset, _ = detector.infer(frame)
        dets = [(p["x"], p["y"], p["score"]) for p in decode_points(hm, offset, args.conf)
                if in_lane(p["x"], p["y"])]
        matched, false_pos, used = score_frame(dets, labels[i])
        tp += len(matched)
        fp += len(false_pos)
        fn += len(labels[i]) - len(used)
        per_frame.append((len(false_pos), i, labels[i], matched, false_pos, used))

    print(f"{args.model} | conf {args.conf} | frame {args.f0}-{args.f1}: {summary(tp, fp, fn)}")

    per_frame.sort(key=lambda t: -t[0])
    for n_fp, i, gts, matched, false_pos, used in per_frame[: args.top]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, img = cap.read()
        for k, (gx, gy) in enumerate(gts):
            cv2.circle(img, (int(gx), int(gy)), 8, (0, 0, 255), -1)
            if k not in used:
                cv2.circle(img, (int(gx), int(gy)), 22, (0, 0, 255), 3)
        for x, y, _ in matched:
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
        for x, y, s in false_pos:
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(img, f"{s:.2f}", (int(x) + 8, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imwrite(f"{args.out}/frame_{i:04d}_palsu{n_fp}.png", img)
    print(f"Gambar disimpan di {args.out}/")


if __name__ == "__main__":
    main()
