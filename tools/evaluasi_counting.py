"""
Evaluasi counting dan klasifikasi terhadap hitungan manual.

Masukan:
  - counting_log.csv  : keluaran run.py (kolom id dan kelas dipakai)
  - berkas kebenaran  : CSV dengan kolom  id,kelas_benar
      * id            : ID tampilan di counting_log.csv, atau kata "terlewat" untuk
                        kendaraan yang melewati garis tetapi tidak terhitung
      * kelas_benar   : car / van / truck, atau "bukan" bila yang terhitung ternyata
                        bukan kendaraan (deteksi palsu) atau bukan kelas yang ditangani

Contoh berkas kebenaran:
  id,kelas_benar
  1,car
  2,car
  3,bukan
  4,truck
  terlewat,van

Keluaran: jumlah per kelas (prediksi vs manual), precision/recall/F1 counting,
dan confusion matrix klasifikasi. Id di log yang tidak ada di berkas kebenaran
dilaporkan sebagai belum dilabeli dan tidak dihitung.

Pakai (dari folder proyek):
  python -m tools.evaluasi_counting --log counting_log.csv --truth kebenaran.csv

Precision, recall, F1: definisi standar (Powers 2011), lihat tools/metrics.py.
"""

import csv
from argparse import ArgumentParser

import config as cfg
from tools.metrics import prf

BUKAN = "bukan"
TERLEWAT = "terlewat"


def read_log(path):
    with open(path, encoding="utf-8") as f:
        return [(r["id"], r["kelas"]) for r in csv.DictReader(f)]


def read_truth(path):
    """Mengembalikan (peta id -> kelas benar, daftar kelas benar kendaraan terlewat)."""
    by_id, missed = {}, []
    valid = set(cfg.CLASSES) | {BUKAN}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key, cls = r["id"].strip(), r["kelas_benar"].strip().lower()
            if key.lower() == TERLEWAT:
                if cls not in cfg.CLASSES:
                    raise SystemExit(f"Kendaraan terlewat harus bertipe {cfg.CLASSES}, ditemukan: {cls!r}")
                missed.append(cls)
                continue
            if cls not in valid:
                raise SystemExit(f"Kelas tidak dikenal untuk id {key}: {cls!r} (pilih {sorted(valid)})")
            if key in by_id:
                raise SystemExit(f"id {key} muncul dua kali di berkas kebenaran")
            by_id[key] = cls
    return by_id, missed


def evaluate(log_rows, truth_by_id, missed):
    """
    log_rows: [(id, kelas prediksi)]; truth_by_id: {id: kelas benar atau 'bukan'};
    missed: [kelas benar kendaraan yang lewat garis tetapi tidak terhitung].
    """
    labels = list(cfg.CLASSES) + [BUKAN]
    conf = {t: {p: 0 for p in cfg.CLASSES} for t in labels}  # baris = benar, kolom = prediksi
    unlabeled = []
    for key, pred in log_rows:
        if key not in truth_by_id:
            unlabeled.append(key)
            continue
        conf[truth_by_id[key]][pred] += 1

    tp = sum(conf[c][p] for c in cfg.CLASSES for p in cfg.CLASSES)  # terhitung dan memang kendaraan
    fp = sum(conf[BUKAN].values())
    fn = len(missed)
    right_class = sum(conf[c][c] for c in cfg.CLASSES)

    pred_total = {c: sum(conf[t][c] for t in labels) for c in cfg.CLASSES}
    true_total = {c: sum(conf[c].values()) + missed.count(c) for c in cfg.CLASSES}
    return {
        "conf": conf,
        "unlabeled": unlabeled,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "prf": prf(tp, fp, fn),
        "class_acc": right_class / tp if tp else 0.0,
        "right_class": right_class,
        "pred_total": pred_total,
        "true_total": true_total,
    }


def report(res):
    print("Jumlah per kelas (prediksi vs manual):")
    print("  kelas   prediksi  manual  selisih")
    for c in cfg.CLASSES:
        p, t = res["pred_total"][c], res["true_total"][c]
        print(f"  {c:6}  {p:8}  {t:6}  {p - t:+7}")
    p_all, t_all = sum(res["pred_total"].values()), sum(res["true_total"].values())
    print(f"  {'total':6}  {p_all:8}  {t_all:6}  {p_all - t_all:+7}")

    p, r, f1 = res["prf"]
    print(
        f"\nCounting: benar {res['tp']}, palsu {res['fp']}, terlewat {res['fn']} | "
        f"precision {p:.3f}, recall {r:.3f}, F1 {f1:.3f}"
    )
    print(
        f"Klasifikasi: {res['right_class']} dari {res['tp']} kendaraan terhitung bertipe benar "
        f"(akurasi {res['class_acc']:.3f})"
    )

    print("\nConfusion matrix (baris = kelas benar, kolom = prediksi):")
    print("  " + " " * 8 + "".join(f"{c:>8}" for c in cfg.CLASSES))
    for t, row in res["conf"].items():
        print(f"  {t:8}" + "".join(f"{row[c]:8}" for c in cfg.CLASSES))
    print(f"  (baris '{BUKAN}' = terhitung tetapi bukan kendaraan; terlewat tidak masuk matriks)")

    if res["unlabeled"]:
        print(f"\nBelum dilabeli ({len(res['unlabeled'])}): id {', '.join(res['unlabeled'])}")


def main():
    ap = ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--log", default=cfg.COUNT_LOG)
    ap.add_argument("--truth", required=True, help="CSV: id,kelas_benar")
    args = ap.parse_args()
    truth_by_id, missed = read_truth(args.truth)
    report(evaluate(read_log(args.log), truth_by_id, missed))


if __name__ == "__main__":
    main()
