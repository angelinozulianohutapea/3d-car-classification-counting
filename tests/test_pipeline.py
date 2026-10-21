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
