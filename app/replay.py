"""
replay.py — precomputes the three trajectories (ground truth, naive, AI-fused)
for each bundled demo trip, using the SAME batch fusion functions validated in
offline_eval.py. Precomputing server-side (rather than recomputing per-request)
keeps /api/replay/{id}/next cheap and avoids re-deriving incremental logic
that would duplicate (and risk diverging from) the tested batch path.
"""
from pathlib import Path

from .data_utils import load_trip
from .geo import LocalFrame
from .fusion import naive_dead_reckoning, ai_fused_dead_reckoning, ground_truth_positions
from .model import get_model_bundle

DATA_DIR = Path(__file__).parent / "replay_data"

TRIPS = {
    "vta01a": {
        "label": "Vta01a — 60s highway blackout (held-out driver)",
        "s_file": DATA_DIR / "vta01a_S.csv",
        "v_file": DATA_DIR / "vta01a_V.csv",
        "warmup": 40,  # samples of GNSS-available warmup before the simulated blackout
    }
}

_cache = {}


def get_trip(trip_id: str):
    if trip_id not in TRIPS:
        raise KeyError(trip_id)
    if trip_id in _cache:
        return _cache[trip_id]

    cfg = TRIPS[trip_id]
    for key in ("s_file", "v_file"):
        if not cfg[key].is_file():
            raise RuntimeError(
                f"Bundled replay data missing: {cfg[key]}. These CSVs must be committed "
                "to the repository for replay mode to work in a deployed environment."
            )
    df = load_trip(str(cfg["s_file"]), str(cfg["v_file"]))
    warmup = cfg["warmup"]
    start_idx = warmup
    end_idx = len(df)

    ref = LocalFrame(df["lat"].values[start_idx], df["lon"].values[start_idx])
    model_bundle = get_model_bundle()

    gt_x, gt_y = ground_truth_positions(df, start_idx, end_idx, ref)
    naive_x, naive_y = naive_dead_reckoning(df, start_idx, end_idx, ref)
    ai_x, ai_y, ai_speed = ai_fused_dead_reckoning(df, start_idx, end_idx, ref, model_bundle)

    warmup_lats = df["lat"].values[:warmup].tolist()
    warmup_lons = df["lon"].values[:warmup].tolist()

    gt_lat, gt_lon = [], []
    naive_lat, naive_lon = [], []
    ai_lat, ai_lon = [], []
    for i in range(len(gt_x)):
        la, lo = ref.to_latlon(gt_x[i], gt_y[i]); gt_lat.append(la); gt_lon.append(lo)
        la, lo = ref.to_latlon(naive_x[i], naive_y[i]); naive_lat.append(la); naive_lon.append(lo)
        la, lo = ref.to_latlon(ai_x[i], ai_y[i]); ai_lat.append(la); ai_lon.append(lo)

    result = {
        "label": cfg["label"],
        "warmup_lat": warmup_lats, "warmup_lon": warmup_lons,
        "gt_lat": gt_lat, "gt_lon": gt_lon,
        "naive_lat": naive_lat, "naive_lon": naive_lon,
        "ai_lat": ai_lat, "ai_lon": ai_lon,
        "ai_speed_kmh": [round(float(s), 1) for s in ai_speed],
        "num_warmup": warmup,
        "num_blackout": len(gt_x),
        "dt": 0.1,
    }
    _cache[trip_id] = result
    return result
