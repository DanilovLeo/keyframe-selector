"""Quick end-to-end smoke test against the real LIBERO demo file."""
import sys, numpy as np
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")

from src.utils.loader import load_libero_demo
from src.extractors import (
    UniformExtractor, VelocityZeroExtractor,
    GripperStateExtractor, AWEExtractor,
)
from src.extractors.base import KeyframeExtractor
from src.eval.metrics import compression_ratio

DEMO_PATH = (
    "~/keyframe_selection/LIBERO/libero/datasets/libero_spatial/"
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
)

demo = load_libero_demo(DEMO_PATH, demo_idx=0)
print("task_name :", demo["task_name"])
print("n_demos   :", demo["n_demos"])
for k, v in demo.items():
    if k not in ("n_demos", "task_name"):
        arr = np.asarray(v)
        print("  %-16s shape=%-22s dtype=%s" % (k, str(arr.shape), arr.dtype))

T = len(demo["ee_pos"])
print("\nT=%d  gripper range=[%.4f, %.4f]" % (
    T, demo["gripper_state"].min(), demo["gripper_state"].max()))

specs = [
    ("uniform",  UniformExtractor(n_keyframes=10),       demo["ee_pos"]),
    ("velocity", VelocityZeroExtractor(threshold=0.02),  demo["ee_vel"]),
    ("gripper",  GripperStateExtractor(min_dist=3),      demo["gripper_state"]),
    ("awe",      AWEExtractor(error_threshold=0.01),     demo["ee_pos"]),
]

print("\n%-12s %4s %7s  Indices" % ("Method", "N", "CR"))
print("-" * 70)
for name, ext, traj in specs:
    kf = ext.extract(traj)
    cr = compression_ratio(T, len(kf))
    print("%-12s %4d %7.3f  %s" % (name, len(kf), cr, kf.tolist()))
    assert kf[0] == 0,          name + ": first index not 0"
    assert kf[-1] == T - 1,     name + ": last index not T-1"
    assert (np.diff(kf) > 0).all(), name + ": indices not sorted"

print("\nAll sanity checks passed.")
