"""
Fine-tune head heatmap (outc) CenterNet dengan label titik dari label_points.py.
Backbone dan head offset/ukuran dibekukan. Tidak ada augmentasi.
Pakai:
  python finetune_heatmap.py 13834567_1280x720.mp4 centernet-3d-bbox.pth --val-from 330
"""
import copy
import json
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
TOL_PX = 48  # deteksi dianggap benar bila dalam jarak ini dari titik label

ap = ArgumentParser()
ap.add_argument("video")
ap.add_argument("model")
ap.add_argument("--labels", default="labels.json")
ap.add_argument("--out", default="centernet-ft.pth")
ap.add_argument("--val-from", type=int, default=None, help="frame >= nilai ini dipakai validasi")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--sigma", type=float, default=0.8, help="lebar Gaussian target (satuan sel)")
args = ap.parse_args()

with open(args.labels, encoding="utf-8") as f:
    labels = {int(k): v["points"] for k, v in json.load(f).items() if v["done"]}
if not labels:
    raise SystemExit("Tidak ada frame berstatus SELESAI di " + args.labels)

model = centernet()
model.load_state_dict(torch.load(args.model, map_location="cpu"))
model.eval()
for p in model.parameters():
    p.requires_grad = False
for p in model.outc.parameters():
    p.requires_grad = True
base_head = copy.deepcopy(model.outc)

prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(args.video)
feats = {}
print(f"Menghitung fitur {len(labels)} frame berlabel ...")
with torch.no_grad():
    for i in sorted(labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Gagal membaca frame {i}")
        feats[i] = model.base_model(prep(frame).float().unsqueeze(0))
cap.release()
hh, ww = feats[next(iter(feats))].shape[-2:]
print(f"Ukuran heatmap: {hh} x {ww} (stride {STRIDE})")


def target_map(points):
    t = np.zeros((hh, ww), dtype=np.float32)
    ys, xs = np.mgrid[0:hh, 0:ww]
    for x, y in points:
        ix = min(max(int(x / STRIDE), 0), ww - 1)
        iy = min(max(int(y / STRIDE), 0), hh - 1)
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * args.sigma ** 2))
        t = np.maximum(t, g.astype(np.float32))
        t[iy, ix] = 1.0
    return torch.from_numpy(t)


def focal_loss(logit, target):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p ** 2 * (1 - target) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / max(pos.sum().item(), 1.0)


def peaks(head, feat, conf):
    with torch.no_grad():
        hm = torch.sigmoid(head(feat))
        keep = F.max_pool2d(hm, 3, stride=1, padding=1) == hm
        hm = (hm * keep)[0, 0].numpy()
    out = []
    for cy, cx in np.argwhere(hm > conf):
        off = (0.5, 0.5)  # offset tidak dipakai di evaluasi; pakai tengah sel
        out.append(((cx + off[0]) * STRIDE, (cy + off[1]) * STRIDE, float(hm[cy, cx])))
    return out


def in_lane(x, y):
    g = get_position_ground(x, y, K, to_world_from_camera, 720)
    return abs(g[0]) <= 11


def evaluate(head, ids, conf):
    tp = fp = fn = 0
    for i in ids:
        gts = [tuple(p) for p in labels[i]]
        dets = [(x, y) for x, y, s in peaks(head, feats[i], conf) if in_lane(x, y)]
        used = set()
        for dx, dy in dets:
            best, bd = None, TOL_PX
            for k, (gx, gy) in enumerate(gts):
                d = np.hypot(dx - gx, dy - gy)
                if k not in used and d <= bd:
                    best, bd = k, d
            if best is None:
                fp += 1
            else:
                used.add(best)
                tp += 1
        fn += len(gts) - len(used)
    return tp, fp, fn


all_ids = sorted(labels)
if args.val_from is None:
    train_ids, val_ids = all_ids, []
else:
    train_ids = [i for i in all_ids if i < args.val_from]
    val_ids = [i for i in all_ids if i >= args.val_from]
print(f"Latih: {len(train_ids)} frame | validasi: {len(val_ids)} frame | "
      f"titik latih: {sum(len(labels[i]) for i in train_ids)}")

targets = {i: target_map(labels[i]) for i in all_ids}
opt = torch.optim.Adam(model.outc.parameters(), lr=args.lr)


def report(tag, head, ids):
    if not ids:
        return
    for conf in (0.2, 0.3, 0.5):
        tp, fp, fn = evaluate(head, ids, conf)
        print(f"  {tag:10} conf {conf}: benar {tp}, palsu {fp}, terlewat {fn}")


print("\nSebelum fine-tuning (model asli):")
report("latih", base_head, train_ids)
report("validasi", base_head, val_ids)

for ep in range(1, args.epochs + 1):
    order = np.random.permutation(train_ids)
    total = 0.0
    for i in order:
        loss = focal_loss(model.outc(feats[i])[0, 0], targets[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    if ep % 10 == 0 or ep == 1:
        print(f"epoch {ep:3}: loss rata-rata {total / len(train_ids):.4f}")

print("\nSetelah fine-tuning:")
report("latih", model.outc, train_ids)
report("validasi", model.outc, val_ids)

torch.save(model.state_dict(), args.out)
print(f"\nModel disimpan: {args.out}")
