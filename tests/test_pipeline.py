"""
Uji sederhana tanpa video dan tanpa PyTorch (data sintetis).
Jalankan dari folder proyek:  python -m tests.test_pipeline
"""

import numpy as np

import config as cfg
from box3d import blended_dims
from classifier import VehicleClassifier, classify_by_length
from counter import LineCounter
from detection_filter import reject_reason
from geometry import get_position_ground, project
from kalman_filter import KalmanFilter
from tests.synthetic import DT, make_stream
from tracker import Tracker


def test_geometry_roundtrip():
    for x, z in [(2.0, 20.0), (-5.0, 12.0), (0.0, 35.0)]:
        u, v_up = project(np.array([x, 0.0, z]))
        g = get_position_ground(u, cfg.IMG_H - v_up)
        assert abs(g[0] - x) < 1e-6 and abs(g[2] - z) < 1e-6 and abs(g[1]) < 1e-6, (x, z, g)


def test_kalman_velocity():
    kf = KalmanFilter([0.0, 30.0, 2.0, 1.5, 4.5], DT)
    for k in range(1, 61):
        kf.predict()
        kf.update([0.0, 30.0 - 20.0 * k * DT, 2.0, 1.5, 4.5])
    assert abs(kf.estimate[2, 0] + 20.0) < 1.0, kf.estimate[2, 0]


def test_classifier():
    assert [classify_by_length(l) for l in (4.5, 5.2, 6.9, 7.0, 10)] == ["car", "van", "van", "truck", "truck"]
    c = VehicleClassifier()
    for l in (4.4, 4.6, 6.0, 4.5):
        c.update(1.8, 1.5, l)
    assert c.cls == "car" and c.votes == {"car": 3, "van": 1, "truck": 0}
    w, h, l = blended_dims(c.cls, c.dim_buf)
    lo, hi = cfg.LIMITS["car"]["l"]
    assert lo <= l <= hi


def test_filter():
    ground = np.array([0.0, 0.0, 20.0])
    ok = {"w": 0.1, "h": 0.1, "l": 0.25, "x": 0, "y": 0}
    assert reject_reason(ok, ground, 20.0) is None
    assert reject_reason(ok, np.array([12.0, 0, 20.0]), 20.0) == "luar jalur"
    assert reject_reason({**ok, "w": 0.01}, ground, 20.0) == "w kecil"
    assert reject_reason({**ok, "h": 0.5}, ground, 20.0) == "tinggi"
    assert reject_reason({**ok, "l": 0.1}, ground, 20.0) == "l pendek"
    assert reject_reason({**ok, "h": 0.2, "l": 0.16}, ground, 20.0) == "h>l"


def _det(y, x=0.0, score=0.9):
    return {"x": x, "y": y, "w": 1.85, "h": 1.5, "l": 4.5, "score": score}


def _drive(tracker, ys, x=0.0, skip=(), **kw):
    """Satu mobil mendekat; frame di `skip` tidak mendapat deteksi."""
    for k, y in enumerate(ys):
        tracker.update([] if k in skip else [_det(y, x)], **kw)
        tracker.assign_display_ids()


def test_tracker_single_vehicle_keeps_one_track():
    tr = Tracker(DT)
    ys = [36.0 - 0.6 * k for k in range(40)]
    ids = set()
    for y in ys:
        tr.update([_det(y)])
        tr.assign_display_ids()
        assert len(tr.tracks) == 1, len(tr.tracks)
        ids.add(tr.tracks[0].id)
    assert len(ids) == 1
    assert tr.tracks[0].display_id == 1  # layak tampil setelah MIN_HITS_TO_DISPLAY


def test_tracker_survives_short_dropout():
    tr = Tracker(DT)
    ys = [36.0 - 0.6 * k for k in range(40)]
    _drive(tr, ys, skip=range(15, 20))  # 5 frame hilang < MAX_LOST_FRAMES
    assert len(tr.tracks) == 1 and tr.tracks[0].id == 0, [t.id for t in tr.tracks]


def test_tracker_drops_lost_track():
    tr = Tracker(DT)
    _drive(tr, [30.0 - 0.6 * k for k in range(10)])
    assert len(tr.tracks) == 1
    for _ in range(cfg.MAX_LOST_FRAMES - 1):
        tr.update([])
    assert len(tr.tracks) == 1  # belum mencapai MAX_LOST_FRAMES
    tr.update([])
    assert tr.tracks == []


def test_tracker_low_score_extends_but_never_creates():
    tr = Tracker(DT)
    _drive(tr, [30.0 - 0.6 * k for k in range(10)], conf_high=0.5)
    n_before, first_id = len(tr.tracks), tr.tracks[0].id
    # deteksi skor rendah di dekat track: menyambung, tidak membuat track baru
    tr.update([_det(30.0 - 0.6 * 10, score=0.3)], conf_high=0.5)
    assert len(tr.tracks) == n_before and tr.tracks[0].id == first_id
    assert tr.tracks[0].unassociated_counter == 0
    # deteksi skor rendah yang jauh dari semua track: diabaikan
    tr.update([_det(12.0, x=6.0, score=0.3)], conf_high=0.5)
    assert len(tr.tracks) == n_before


def test_tracker_merges_duplicates():
    tr = Tracker(DT)
    # dua deteksi hampir berimpit di frame pertama -> dua track, lalu satu dibuang
    tr.update([_det(25.0, x=0.0), _det(25.5, x=0.5)])
    assert len(tr.tracks) == 2
    tr.update([_det(24.4, x=0.0), _det(24.9, x=0.5)])
    assert len(tr.tracks) == 1, len(tr.tracks)


def test_tracker_two_vehicles_stay_separate():
    tr = Tracker(DT)
    for k in range(30):
        tr.update([_det(36.0 - 0.6 * k, x=-5.0), _det(30.0 - 0.4 * k, x=5.0)])
        tr.assign_display_ids()
    assert len(tr.tracks) == 2
    assert {t.display_id for t in tr.tracks} == {1, 2}


def test_evaluasi_counting():
    from tools.evaluasi_counting import evaluate

    log = [("1", "car"), ("2", "car"), ("3", "truck"), ("4", "van"), ("5", "car")]
    truth = {"1": "car", "2": "van", "3": "bukan", "4": "van"}  # id 5 belum dilabeli
    res = evaluate(log, truth, missed=["truck"])
    assert (res["tp"], res["fp"], res["fn"]) == (3, 1, 1)
    assert res["right_class"] == 2 and abs(res["class_acc"] - 2 / 3) < 1e-9
    assert res["unlabeled"] == ["5"]
    assert res["true_total"] == {"car": 1, "van": 2, "truck": 1}
    p, r, f1 = res["prf"]
    assert abs(p - 0.75) < 1e-9 and abs(r - 0.75) < 1e-9 and abs(f1 - 0.75) < 1e-9


def run(conf_high):
    frames, vehicles = make_stream()
    tracker, counter = Tracker(DT), LineCounter(15.0)
    for idx, dets in enumerate(frames):
        tracker.update(dets, conf_high=conf_high)
        tracker.assign_display_ids()
        counter.update(tracker.tracks, idx)
    return counter, vehicles


def test_counting():
    for conf_high in (None, 0.5):
        counter, vehicles = run(conf_high)
        truth = {c: sum(v["kind"] == c for v in vehicles) for c in cfg.CLASSES}
        assert counter.total() == len(vehicles), (conf_high, counter.counts)
        assert counter.counts == truth, (conf_high, counter.counts, truth)
        ids = [e[1] for e in counter.events]
        assert len(ids) == len(set(ids))  # tiap track dihitung sekali


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK  ", name)
    print("Semua uji lulus")
