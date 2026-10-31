"""
Fine-tuning head heatmap (outc) CenterNet dengan label titik dari label_points.

Backbone serta head offset dan ukuran dibekukan. Tidak ada augmentasi.
Fungsi kerugian: focal loss versi CenterNet, Zhou dkk. 2019 (arXiv:1904.07850)
persamaan 1 (alpha = 2, beta = 4), mengacu pada Lin dkk., "Focal Loss for Dense
Object Detection", ICCV 2017. Lebar Gaussian target tetap (--sigma) [heuristik];
CenterNet memakai radius yang bergantung ukuran objek.

Setiap eksekusi menghasilkan model BARU berversi (tidak menimpa model lama):
  - centernet-ft_TANGGAL-JAM.pth        bobot hasil fine-tuning (epoch terbaik menurut
                                        F1 validasi bila --val-from dipakai; selain itu epoch terakhir)
  - centernet-ft_TANGGAL-JAM_perbandingan.csv   tabel model asli vs model baru
  - riwayat_model.csv                   satu baris per eksekusi (ditambah, tidak ditimpa)

Pakai (dari folder proyek):
  python -m tools.finetune_heatmap VIDEO.mp4 centernet-3d-bbox.pth --val-from 330
  python -m tools.finetune_heatmap VIDEO.mp4 centernet-3d-bbox.pth --val-from 330 --out model-v2.pth
"""

import copy
import csv
import json
import os
from argparse import ArgumentParser
from datetime import datetime

import cv2
import numpy as np
import torch

import config as cfg
from detector import build_preprocess, decode_points, load_model
from tools.metrics import TOL_PX, in_lane, prf, score_frame, summary

STRIDE = cfg.MODEL_SCALE
EVAL_CONFS = (0.2, 0.3, 0.5)
BEST_CONF = 0.3  # ambang untuk memilih epoch terbaik dan untuk riwayat
EVAL_EVERY = 10  # epoch

CSV_HEADER = ["model", "himpunan", "conf", "benar", "palsu", "terlewat", "precision", "recall", "F1"]
HISTORY_HEADER = ["waktu", "berkas_model", "model_awal", "video", "labels", "epoch", "epoch_terbaik",
                  "lr", "sigma", "val_from", "himpunan_acuan", "F1_asli", "F1_baru"]


def unique_path(path):
    """Jangan menimpa berkas yang sudah ada: tambahkan _2, _3, ... bila perlu."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 2
    while os.path.exists(f"{stem}_{k}{ext}"):
        k += 1
    return f"{stem}_{k}{ext}"


def write_comparison_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)


def append_history(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HISTORY_HEADER)
        w.writerow(row)


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
    ap.add_argument("--out", default=None,
                    help="berkas .pth keluaran (bawaan: centernet-ft_TANGGAL-JAM.pth; berkas lama tidak ditimpa)")
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

    def report(model_name, tag, head, ids, rows):
        """Cetak dan catat ke rows (untuk CSV) hasil evaluasi pada beberapa ambang."""
        if not ids:
            return
        for conf in EVAL_CONFS:
            tp, fp, fn = evaluate(head, ids, conf)
            print(f"  {tag:10} conf {conf}: {summary(tp, fp, fn)}")
            p, r, f1 = prf(tp, fp, fn)
            rows.append([model_name, tag, conf, tp, fp, fn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    def f1_at(head, ids, conf=BEST_CONF):
        return prf(*evaluate(head, ids, conf))[2]

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

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = unique_path(args.out or f"centernet-ft_{stamp}.pth")
    csv_path = os.path.splitext(out)[0] + "_perbandingan.csv"
    rows = []

    print("\nSebelum fine-tuning (model asli):")
    report("asli", "latih", base_head, train_ids, rows)
    report("asli", "validasi", base_head, val_ids, rows)

    ref_ids, ref_name = (val_ids, "validasi") if val_ids else (train_ids, "latih")
    best_f1, best_ep, best_state = -1.0, 0, None

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
        # Pilih epoch terbaik hanya bila ada data validasi terpisah; tanpa itu
        # memilih berdasarkan data latih tidak bermakna, jadi pakai epoch terakhir.
        if val_ids and (ep % EVAL_EVERY == 0 or ep == args.epochs):
            f1 = f1_at(model.outc, val_ids)
            print(f"  F1 validasi (conf {BEST_CONF}) epoch {ep}: {f1:.3f}")
            if f1 > best_f1:
                best_f1, best_ep = f1, ep
                best_state = copy.deepcopy(model.outc.state_dict())

    if best_state is not None:
        model.outc.load_state_dict(best_state)
        print(f"\nEpoch terbaik: {best_ep} (F1 validasi {best_f1:.3f}); bobot itulah yang disimpan.")
    else:
        best_ep = args.epochs

    print("\nSetelah fine-tuning (model baru):")
    report("baru", "latih", model.outc, train_ids, rows)
    report("baru", "validasi", model.outc, val_ids, rows)

    torch.save(model.state_dict(), out)
    write_comparison_csv(csv_path, rows)
    f1_old = f1_at(base_head, ref_ids)
    f1_new = f1_at(model.outc, ref_ids)
    append_history(os.path.join(os.path.dirname(os.path.abspath(out)), "riwayat_model.csv"), [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), os.path.basename(out), os.path.basename(args.model),
        os.path.basename(args.video), os.path.basename(args.labels), args.epochs, best_ep,
        args.lr, args.sigma, args.val_from if args.val_from is not None else "",
        ref_name, f"{f1_old:.3f}", f"{f1_new:.3f}",
    ])
    print(f"\nModel disimpan      : {out}")
    print(f"Tabel perbandingan  : {csv_path}")
    print(f"Perbandingan F1 ({ref_name}, conf {BEST_CONF}): asli {f1_old:.3f} -> baru {f1_new:.3f}")


if __name__ == "__main__":
    main()