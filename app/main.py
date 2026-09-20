"""
main.py — TrueTrack backend.

Endpoints:
  GET  /api/health
  GET  /api/replay/trips                 -> list bundled demo trips
  POST /api/replay/{trip_id}/start       -> {session_id, ...trip metadata}
  POST /api/replay/session/{sid}/next    -> advance one 100ms step, body {simulate_outage: bool}
  POST /api/live/session/start           -> {session_id}
  POST /api/live/session/{sid}/sample    -> feed one live IMU(+GPS) sample, get fused position back

Run locally:   uvicorn app.main:app --reload --port 8000
Then open:     http://localhost:8000/
"""
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .replay import get_trip, TRIPS
from .model import get_model_bundle
from .fusion import OnlineFusionSession

app = FastAPI(title="TrueTrack")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for a hackathon demo; tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"

# ---------------------------------------------------------------------------
# Replay mode (feeds a real, pre-recorded trip -- lets you demo GNSS-denied
# navigation indoors, without a moving vehicle, per the PS's own suggestion)
# ---------------------------------------------------------------------------

_replay_sessions = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/replay/trips")
def list_trips():
    return [{"id": tid, "label": cfg["label"]} for tid, cfg in TRIPS.items()]


@app.post("/api/replay/{trip_id}/start")
def start_replay(trip_id: str):
    try:
        trip = get_trip(trip_id)
    except KeyError:
        raise HTTPException(404, f"Unknown trip '{trip_id}'")
    except RuntimeError as exc:
        # model failed to load, or the bundled replay CSVs are missing
        raise HTTPException(503, str(exc))

    session_id = str(uuid.uuid4())
    _replay_sessions[session_id] = {"trip": trip, "index": -1}
    return {
        "session_id": session_id,
        "label": trip["label"],
        "num_warmup": trip["num_warmup"],
        "num_blackout": trip["num_blackout"],
        "dt": trip["dt"],
    }


class ReplayStepRequest(BaseModel):
    simulate_outage: bool = True


@app.post("/api/replay/session/{sid}/next")
def replay_next(sid: str, req: ReplayStepRequest):
    sess = _replay_sessions.get(sid)
    if sess is None:
        raise HTTPException(404, "Unknown session")

    trip = sess["trip"]
    sess["index"] += 1
    i = sess["index"]

    total = trip["num_warmup"] + trip["num_blackout"]
    if i >= total:
        return {"done": True}

    if i < trip["num_warmup"]:
        # still in the GNSS-available warmup phase
        return {
            "done": False, "phase": "warmup", "mode": "GNSS",
            "lat": trip["warmup_lat"][i], "lon": trip["warmup_lon"][i],
        }

    b = i - trip["num_warmup"]
    out = {
        "done": False, "phase": "blackout",
        "ground_truth": {"lat": trip["gt_lat"][b], "lon": trip["gt_lon"][b]},
    }
    if req.simulate_outage:
        out["mode"] = "DR"
        out["ai_fused"] = {"lat": trip["ai_lat"][b], "lon": trip["ai_lon"][b]}
        out["naive"] = {"lat": trip["naive_lat"][b], "lon": trip["naive_lon"][b]}
        out["speed_kmh"] = trip["ai_speed_kmh"][b]
    else:
        # outage toggle off mid-blackout -> just show ground truth (GNSS "restored")
        out["mode"] = "GNSS"
        out["lat"] = trip["gt_lat"][b]
        out["lon"] = trip["gt_lon"][b]
    return out


# ---------------------------------------------------------------------------
# Live mode (real phone sensors via the browser)
# ---------------------------------------------------------------------------

_live_sessions = {}


@app.post("/api/live/session/start")
def start_live_session():
    try:
        bundle = get_model_bundle()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    session_id = str(uuid.uuid4())
    _live_sessions[session_id] = OnlineFusionSession(bundle)
    return {"session_id": session_id}


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Gyro(BaseModel):
    yaw: float
    pitch: float
    roll: float


class GpsFix(BaseModel):
    lat: float
    lon: float


class LiveSample(BaseModel):
    accel: Vec3
    gravity: Vec3
    gyro: Gyro
    gps: Optional[GpsFix] = None
    simulate_outage: bool = False
    dt: float = 0.1


@app.post("/api/live/session/{sid}/sample")
def live_sample(sid: str, sample: LiveSample):
    sess = _live_sessions.get(sid)
    if sess is None:
        raise HTTPException(404, "Unknown session -- call /api/live/session/start first")

    gps = {"lat": sample.gps.lat, "lon": sample.gps.lon} if sample.gps else None
    result = sess.update(
        accel=sample.accel.model_dump(), gravity=sample.gravity.model_dump(),
        gyro=sample.gyro.model_dump(), gps=gps,
        simulate_outage=sample.simulate_outage, dt=sample.dt,
    )
    return result


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(404, "Frontend not found (static/index.html is missing)")
    return FileResponse(str(index_file))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # browsers request this on every page load; return 204 rather than a 404 trace
    return Response(status_code=204)
