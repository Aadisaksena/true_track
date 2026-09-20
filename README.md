# TrueTrack — backend

AI-assisted dead reckoning for GNSS-denied vehicle navigation. FastAPI backend
plus a small Leaflet frontend, with two demo modes:

- **Replay** — steps through a real, held-out IO-VNBD trip and compares ground
  truth vs naive double-integration vs the AI-fused estimate.
- **Live** — feeds real phone IMU/GPS samples from the browser through the same
  fusion logic.

---

## Deploying to Render

The repo is configured as a Render Blueprint. Two options:

### Option A — Blueprint (recommended)

1. Push this folder to the root of your GitHub repo.
2. In Render: **New → Blueprint**, pick the repo. It reads `render.yaml`.
3. Deploy. Health check is `GET /api/health`.

### Option B — manual Web Service

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Build command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1` |
| Health check path | `/api/health` |
| Env var | `PYTHON_VERSION` = `3.12.8` |

Once live, the frontend is served at `/` and the API under `/api/...`.

---

## Two things that will break the deploy if you change them

**1. Do not downgrade `numpy` or `scikit-learn`.**
`app/trained_model.joblib` was pickled with scikit-learn 1.8.x on numpy 2.x.
Installing numpy 1.x raises `ModuleNotFoundError: No module named 'numpy._core'`
at the first prediction; scikit-learn < 1.7 fails to reconstruct the
`HistGradientBoostingRegressor`. `requirements.txt` has floors that prevent both.

**2. Keep `--workers 1`.**
Replay and live sessions live in per-process dictionaries. A second worker
would return `404 Unknown session` for roughly half of all requests.

---

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/>.

Live mode uses the browser's motion and geolocation APIs, which browsers only
expose over HTTPS or on `localhost` — it works on Render (HTTPS) and on
localhost, but not over plain HTTP on a LAN IP.

For the training/eval scripts, install the extras too:

```bash
pip install -r requirements-dev.txt
python train_model.py     # rewrites app/trained_model.joblib
python offline_eval.py    # writes drift_comparison.png
```

If you retrain, commit the regenerated `app/trained_model.joblib` and make sure
the versions in `requirements.txt` still match what you trained with.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | liveness probe |
| GET | `/api/replay/trips` | list bundled demo trips |
| POST | `/api/replay/{trip_id}/start` | open a replay session |
| POST | `/api/replay/session/{sid}/next` | advance one 100 ms step |
| POST | `/api/live/session/start` | open a live sensor session |
| POST | `/api/live/session/{sid}/sample` | submit one IMU(+GPS) sample |

---

## Layout

```
app/                  FastAPI app, fusion engine, trained model
  main.py             routes
  fusion.py           naive DR, AI-fused DR, OnlineFusionSession
  data_utils.py       IO-VNBD loading + feature engineering
  geo.py              local ENU frame
  model.py            lazy model loader
  replay_data/        bundled demo trip CSVs (required at runtime)
  trained_model.joblib
static/               Leaflet frontend
training_data/        raw IO-VNBD trips (training only, ~70 MB, not used at runtime)
train_model.py        retrains the velocity model
offline_eval.py       blackout drift evaluation -> drift_comparison.png
```

`training_data/` is only read by `train_model.py` and `offline_eval.py`. If you
want faster clones and builds you can move it out of the repo entirely; the
deployed service does not touch it.

---

## What was changed to make this deployable

- **`requirements.txt`** — `numpy==1.26.4` and `scikit-learn==1.5.1` were
  incompatible with the committed model (trained on numpy 2.x / sklearn 1.8.x).
  Replaced exact pins with correct bounded ranges. This was the actual deploy
  breaker.
- **`runtime.txt`** — had no trailing newline, so the version string could be
  misparsed and the build would fall back to a newer Python with no wheels for
  the pinned numpy. Fixed, and added `.python-version` (Render's current
  mechanism) pinning 3.12.8.
- **`render.yaml`** — `env: python` is deprecated, replaced with
  `runtime: python`; added `healthCheckPath`, `PYTHON_VERSION`, and explicit
  `--workers 1`.
- **`app/model.py`** — load is lazy and thread-safe so the process binds `$PORT`
  immediately; dependency-mismatch and missing-file cases now raise readable
  errors instead of an opaque stack trace.
- **`app/main.py`** — `.dict()` → `.model_dump()` (removed in pydantic v3);
  model/data failures return `503` with a real message instead of `500`; static
  mount guarded; added a `favicon.ico` handler.
- **`app/replay.py`** — explicit check that the bundled replay CSVs are present.
- **`.gitignore`** added; committed `__pycache__/*.pyc` removed (they were built
  for Python 3.13 and are stale against the pinned 3.12).
- **`requirements-dev.txt`** — moved `matplotlib` out of the production install.
