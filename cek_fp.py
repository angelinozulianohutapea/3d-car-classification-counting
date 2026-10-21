"""
Evaluasi titik dengan offset (sama seperti pred2box) dan simpan frame dengan deteksi palsu terbanyak.
Pakai: python cek_fp.py VIDEO MODEL --conf 0.3 --from 330 --to 472 --out cek_base
"""
import json
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from object_detector import centernet
from multi_object_tracking import K, to_world_from_camera
from utils import get_position_ground

STRIDE = 32
TOL_PX = 48

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("model")
ap.add_argument("--labels", default="labels.json")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--from", dest="f0", type=int, default=330)
ap.add_argument("--to", dest="f1", type=int, default=472)
ap.add_argument("--top", type=int, default=8)
ap.add_argument("--out", default="cek_fp")
args = ap.parse_args()

with open(args.labels, encoding="utf-8") as f:
    labels = {int(k): v["points"] for k, v in json.load(f).items() if v["done"]}

model = centernet()
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()
prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def in_lane(x, y):
    g = get_position_ground(x, y, K, to_world_from_camera, 720)
    return abs(g[0]) <= 11


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
    with torch.no_grad():
        hm, off, _ = model(prep(frame).float().unsqueeze(0))
        hm = torch.sigmoid(hm)
        keep = F.max_pool2d(hm, 3, stride=1, padding=1) == hm
        hm = (hm * keep)[0, 0].numpy()
        off = off[0].numpy()
    dets = []
    for cy, cx in np.argwhere(hm > args.conf):
        x = (cx + off[0, cy, cx]) * STRIDE
        y = (cy + off[1, cy, cx]) * STRIDE
        if in_lane(x, y):
            dets.append((x, y, float(hm[cy, cx])))
    gts = labels[i]
    used, matched, fps = set(), [], []
    for x, y, s in dets:
        best, bd = None, TOL_PX
        for k, (gx, gy) in enumerate(gts):
            d = np.hypot(x - gx, y - gy)
            if k not in used and d <= bd:
                best, bd = k, d
        if best is None:
            fps.append((x, y, s))
        else:
            used.add(best)
            matched.append((x, y, s))
    tp += len(matched)
    fp += len(fps)
    fn += len(gts) - len(used)
    per_frame.append((len(fps), i, gts, matched, fps, used))

print(f"{args.model} | conf {args.conf} | frame {args.f0}-{args.f1}: benar {tp}, palsu {fp}, terlewat {fn}")
per_frame.sort(key=lambda t: -t[0])
for n_fp, i, gts, matched, fps, used in per_frame[: args.top]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    _, img = cap.read()
    for k, (gx, gy) in enumerate(gts):
        cv2.circle(img, (int(gx), int(gy)), 8, (0, 0, 255), -1)
        if k not in used:
            cv2.circle(img, (int(gx), int(gy)), 22, (0, 0, 255), 3)
    for x, y, s in matched:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y, s in fps:
        cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
        cv2.putText(img, f"{s:.2f}", (int(x) + 8, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.imwrite(f"{args.out}/frame_{i:04d}_palsu{n_fp}.png", img)
print(f"Gambar disimpan di {args.out}/")
