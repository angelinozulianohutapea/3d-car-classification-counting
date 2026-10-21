"""
Fine-tuning head heatmap (outc) CenterNet dengan label titik dari label_points.

Backbone serta head offset dan ukuran dibekukan. Tidak ada augmentasi.
Fungsi kerugian: focal loss versi CenterNet, Zhou dkk. 2019 (arXiv:1904.07850)
persamaan 1 (alpha = 2, beta = 4), mengacu pada Lin dkk., "Focal Loss for Dense
Object Detection", ICCV 2017. Lebar Gaussian target tetap (--sigma) [heuristik];
CenterNet memakai radius yang bergantung ukuran objek.

Pakai (dari folder proyek):
  python -m tools.finetune_heatmap VIDEO.mp4 centernet-3d-bbox.pth --val-from 330
"""

import copy
import json
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

import config as cfg
from detector import build_preprocess, decode_points, load_model
from tools.metrics import TOL_PX, in_lane, score_frame, summary

STRIDE = cfg.MODEL_SCALE


def target_map(points, hh, ww, sigma):
    """Target heatmap: Gaussian di tiap titik label, puncak = 1."""
    t = np.zeros((hh, ww), dtype=np.float32)
    ys, xs = np.mgrid[0:hh, 0:ww]
    for x, y in points:
        ix = min(max(int(x / STRIDE), 0), ww - 1)
        iy = min(max(int(y / STRIDE), 0), hh - 1)
        g = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
        t = np.maximum(t, g.astype(np.float32))
        t[iy, ix] = 1.0
    return torch.from_numpy(t)


def focal_loss(logit, target):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(p) * (1 - p) ** 2 * pos
    neg_loss = torch.log(1 - p) * p**2 * (1 - target) ** 4 * neg
    return -(pos_loss.sum() + neg_loss.sum()) / max(pos.sum().item(), 1.0)


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
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

    model = load_model(args.model)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.outc.parameters():
        p.requires_grad = True
    base_head = copy.deepcopy(model.outc)

    prep = build_preprocess()
    cap = cv2.VideoCapture(args.video)
    feats, offsets = {}, {}
    print(f"Menghitung fitur {len(labels)} frame berlabel ...")
    with torch.no_grad():
        for i in sorted(labels):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Gagal membaca frame {i}")
            feats[i] = model.base_model(prep(frame).float().unsqueeze(0))
            offsets[i] = model.outo(feats[i])[0].numpy()  # beku, dipakai untuk dekode evaluasi
    cap.release()
    hh, ww = feats[next(iter(feats))].shape[-2:]
    print(f"Ukuran heatmap: {hh} x {ww} (stride {STRIDE})")

    def evaluate(head, ids, conf):
        tp = fp = fn = 0
        for i in ids:
            with torch.no_grad():
                hm = torch.sigmoid(head(feats[i]))[0, 0].numpy()
            dets = [(p["x"], p["y"], p["score"]) for p in decode_points(hm, offsets[i], conf)
                    if in_lane(p["x"], p["y"])]
            matched, false_pos, used = score_frame(dets, labels[i])
            tp += len(matched)
            fp += len(false_pos)
            fn += len(labels[i]) - len(used)
        return tp, fp, fn

    def report(tag, head, ids):
        if not ids:
            return
        for conf in (0.2, 0.3, 0.5):
            print(f"  {tag:10} conf {conf}: {summary(*evaluate(head, ids, conf))}")

    all_ids = sorted(labels)
    if args.val_from is None:
        train_ids, val_ids = all_ids, []
    else:
        train_ids = [i for i in all_ids if i < args.val_from]
        val_ids = [i for i in all_ids if i >= args.val_from]
    print(f"Latih: {len(train_ids)} frame | validasi: {len(val_ids)} frame | "
          f"titik latih: {sum(len(labels[i]) for i in train_ids)} | toleransi {TOL_PX} px")

    targets = {i: target_map(labels[i], hh, ww, args.sigma) for i in all_ids}
    opt = torch.optim.Adam(model.outc.parameters(), lr=args.lr)

    print("\nSebelum fine-tuning (model asli):")
    report("latih", base_head, train_ids)
    report("validasi", base_head, val_ids)

    for ep in range(1, args.epochs + 1):
        total = 0.0
        for i in np.random.permutation(train_ids):
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


if __name__ == "__main__":
    main()
