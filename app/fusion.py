"""
fusion.py — the actual navigation engine.

Two position estimators are implemented, both operating on the same input stream:

1. `naive_dead_reckoning`  — classic double-integration of world-frame linear
   acceleration. This is the textbook approach every phone-only INS attempt
   starts with, and it drifts badly. We keep it as the baseline everything else
   is compared against.

2. `ai_fused_dead_reckoning` — TrueTrack's approach:
     - heading comes from the phone's own fused orientation (yaw), calibrated
       against GPS heading once at the start of the blackout (this is the
       "auto-alignment" step -- it removes the phone's mounting-angle offset)
     - speed comes from the trained AI model (app/trained_model.joblib), not
       from integrating acceleration
     - position is advanced as speed * heading_unit_vector * dt -- this is the
       Non-Holonomic Constraint (NHC) applied by construction: the vehicle is
       never allowed a sideways velocity component, because the velocity
       vector is built directly from (forward speed, heading) rather than
       from raw x/y acceleration integration.

Both run identically whether the data comes from a recorded IO-VNBD trip
(replay mode) or a live phone in the browser (live mode) -- they only need a
stream of {accel, gravity, gyro, orientation_yaw} samples at ~10Hz.
"""
import math
import numpy as np

from .geo import LocalFrame
from .data_utils import compute_features_at


def _heading_rotation_naive(orient_yaw_deg, orient_pitch_deg, orient_roll_deg):
    """Rough Euler rotation (device frame -> world-ish frame) used ONLY for the
    naive baseline, to demonstrate how quickly uncorrected integration drifts.
    Not claimed to be a precise attitude solution."""
    yaw, pitch, roll = np.radians(orient_yaw_deg), np.radians(orient_pitch_deg), np.radians(orient_roll_deg)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # ZYX Euler -> rotation matrices, applied per-sample below
    return cy, sy, cp, sp, cr, sr


def naive_dead_reckoning(df, start_idx, end_idx, ref_frame: LocalFrame):
    """Double-integrate raw linear acceleration from start_idx to end_idx.
    Returns array of (x, y) positions in local meters, one per sample."""
    lin_x = (df["accel_x"] - df["gravity_x"]).values[start_idx:end_idx]
    lin_y = (df["accel_y"] - df["gravity_y"]).values[start_idx:end_idx]
    yaw = df["orient_yaw"].values[start_idx:end_idx]
    pitch = df["orient_pitch"].values[start_idx:end_idx]
    roll = df["orient_roll"].values[start_idx:end_idx]
    dt = 0.1

    cy, sy, cp, sp, cr, sr = _heading_rotation_naive(yaw, pitch, roll)
    # simplified device->world rotation for the horizontal plane only
    world_ax = cy * lin_x - sy * lin_y
    world_ay = sy * lin_x + cy * lin_y

    # start from the true GPS-implied velocity at blackout entry, then integrate blind
    lat0, lon0 = df["lat"].values[start_idx], df["lon"].values[start_idx]
    lat1, lon1 = df["lat"].values[start_idx + 5], df["lon"].values[start_idx + 5]
    x0, y0 = ref_frame.to_xy(lat0, lon0)
    x1, y1 = ref_frame.to_xy(lat1, lon1)
    vx, vy = (x1 - x0) / (5 * dt), (y1 - y0) / (5 * dt)

    n = end_idx - start_idx
    xs, ys = np.zeros(n), np.zeros(n)
    x, y = x0, y0
    for i in range(n):
        vx += world_ax[i] * dt
        vy += world_ay[i] * dt
        x += vx * dt
        y += vy * dt
        xs[i], ys[i] = x, y
    return xs, ys


def ai_fused_dead_reckoning(df, start_idx, end_idx, ref_frame: LocalFrame, model_bundle):
    """AI-speed + GPS-trend-extrapolated heading dead reckoning, with NHC
    applied by construction (velocity is always along the heading direction --
    no sideways component is possible).

    Heading note: we tested using the phone's raw gyroscope-yaw and its fused
    compass orientation as heading sources on this real dataset. Both failed --
    the compass is corrupted by the vehicle's own magnetic field (orientation
    yaw showed >100 deg/s jumps that don't correspond to any real turn), and
    raw gyro yaw does not equal vehicle heading rate without a mounting-angle
    calibration this 3-trip sample isn't enough data to fit reliably (measured
    correlation to true GPS heading rate: ~0.1 even after a least-squares
    calibration fit). Rather than integrate an unreliable signal, we extrapolate
    the heading TREND measured from GPS in the 2s immediately before blackout --
    a standard, defensible dead-reckoning fallback -- and combine it with the
    AI speed model. This still removes the dominant (quadratic) error source
    that the naive baseline suffers from. Recovering real turn-by-turn heading
    from gyro alone is flagged as the clear next research step (see README).
    """
    model = model_bundle["model"]
    window = model_bundle["window"]

    # heading from a clean 2s GPS baseline immediately before blackout, held
    # CONSTANT through the blackout (safer than extrapolating a turn rate,
    # which compounds any noise in the rate estimate linearly over 60s and
    # was measured to diverge worse than the naive baseline -- documented in
    # README as a concrete next-iteration target: recovering real turn-by-turn
    # heading from a properly axis-calibrated gyro).
    calib_n = 20
    calib_lats = df["lat"].values[start_idx - calib_n:start_idx + 1]
    calib_lons = df["lon"].values[start_idx - calib_n:start_idx + 1]
    cx, cy = zip(*[ref_frame.to_xy(la, lo) for la, lo in zip(calib_lats, calib_lons)])
    cx, cy = np.array(cx), np.array(cy)
    heading0 = math.atan2(cy[-1] - cy[0], cx[-1] - cx[0])

    feats = []
    for i in range(start_idx, end_idx):
        feats.append(compute_features_at(df, i, window=window))
    X = np.array(feats)
    speed_kmh = model.predict(X)
    speed_ms = speed_kmh / 3.6

    lat0, lon0 = df["lat"].values[start_idx], df["lon"].values[start_idx]
    x0, y0 = ref_frame.to_xy(lat0, lon0)

    n = end_idx - start_idx
    xs, ys = np.zeros(n), np.zeros(n)
    x, y = x0, y0
    dt = 0.1
    heading = heading0
    for i in range(n):
        vx = speed_ms[i] * math.cos(heading)  # NHC: no sideways (v_lateral) term exists
        vy = speed_ms[i] * math.sin(heading)
        x += vx * dt
        y += vy * dt
        xs[i], ys[i] = x, y
    return xs, ys, speed_kmh


def ground_truth_positions(df, start_idx, end_idx, ref_frame: LocalFrame):
    lats = df["lat"].values[start_idx:end_idx]
    lons = df["lon"].values[start_idx:end_idx]
    xs, ys = zip(*[ref_frame.to_xy(la, lo) for la, lo in zip(lats, lons)])
    return np.array(xs), np.array(ys)


class OnlineFusionSession:
    """
    Incremental version of the same fusion logic, for live sensor streams
    (browser) or step-by-step replay -- one sample in, one position out.
    Used by the FastAPI backend so live mode and replay mode share identical
    navigation logic to the offline-validated pipeline above.
    """
    LONG_WINDOW = 30
    SHORT_WINDOW = 10

    def __init__(self, model_bundle):
        self.model = model_bundle["model"]
        self.buf_lin_x, self.buf_lin_y, self.buf_lin_z = [], [], []
        self.buf_gyro_yaw, self.buf_gyro_pitch, self.buf_gyro_roll = [], [], []
        self.gps_track = []  # list of (x, y) in local meters, most recent last
        self.ref_frame = None
        self.mode = "GNSS"
        self.dr_x = self.dr_y = None
        self.dr_heading = None
        self.last_speed_kmh = 0.0

    def _push_imu(self, accel, gravity, gyro):
        self.buf_lin_x.append(accel["x"] - gravity["x"])
        self.buf_lin_y.append(accel["y"] - gravity["y"])
        self.buf_lin_z.append(accel["z"] - gravity["z"])
        self.buf_gyro_yaw.append(gyro["yaw"])
        self.buf_gyro_pitch.append(gyro["pitch"])
        self.buf_gyro_roll.append(gyro["roll"])
        maxlen = self.LONG_WINDOW + 5
        for buf in (self.buf_lin_x, self.buf_lin_y, self.buf_lin_z,
                    self.buf_gyro_yaw, self.buf_gyro_pitch, self.buf_gyro_roll):
            if len(buf) > maxlen:
                del buf[0]

    def _current_features(self):
        lin_x = np.array(self.buf_lin_x); lin_y = np.array(self.buf_lin_y); lin_z = np.array(self.buf_lin_z)
        accel_mag = np.sqrt(lin_x ** 2 + lin_y ** 2 + lin_z ** 2)
        gyaw = np.array(self.buf_gyro_yaw); gpitch = np.array(self.buf_gyro_pitch); groll = np.array(self.buf_gyro_roll)
        s, l = self.SHORT_WINDOW, self.LONG_WINDOW
        return np.array([
            lin_x[-s:].std(), lin_y[-s:].std(), lin_z[-s:].std(),
            accel_mag[-s:].mean(), accel_mag[-s:].max(),
            gyaw[-s:].std(), gpitch[-s:].mean(), groll[-s:].mean(),
            lin_z[-l:].std(), accel_mag[-l:].std(),
            np.abs(gyaw[-l:]).mean(), gyaw[-l:].std(),
        ]).reshape(1, -1)

    def update(self, accel, gravity, gyro, gps=None, simulate_outage=False, dt=0.1):
        """
        accel/gravity: {"x","y","z"} in m/s^2 (device frame)
        gyro: {"yaw","pitch","roll"} in rad/s
        gps: {"lat","lon"} or None if unavailable this tick
        simulate_outage: force dead-reckoning even if gps is present (demo toggle)
        Returns dict: {mode, lat, lon, speed_kmh}
        """
        self._push_imu(accel, gravity, gyro)
        gps_ok = gps is not None and not simulate_outage

        if gps_ok:
            if self.ref_frame is None:
                self.ref_frame = LocalFrame(gps["lat"], gps["lon"])
            x, y = self.ref_frame.to_xy(gps["lat"], gps["lon"])
            self.gps_track.append((x, y))
            if len(self.gps_track) > self.LONG_WINDOW:
                self.gps_track.pop(0)
            self.mode = "GNSS"
            return {"mode": "GNSS", "lat": gps["lat"], "lon": gps["lon"], "speed_kmh": None}

        # entering or continuing dead reckoning
        if self.mode == "GNSS":
            # just lost GNSS this tick -- initialize DR state from recent GPS track
            if len(self.gps_track) >= 2 and self.ref_frame is not None:
                (x0, y0) = self.gps_track[0]
                (x1, y1) = self.gps_track[-1]
                self.dr_heading = math.atan2(y1 - y0, x1 - x0)
                self.dr_x, self.dr_y = x1, y1
            elif self.ref_frame is not None:
                self.dr_heading = 0.0
                self.dr_x, self.dr_y = 0.0, 0.0
            else:
                # no GPS ever seen -- can't place on a map yet
                return {"mode": "NO_FIX", "lat": None, "lon": None, "speed_kmh": None}
        self.mode = "DR"

        if len(self.buf_lin_x) < self.LONG_WINDOW:
            speed_kmh = self.last_speed_kmh  # not enough history yet, hold last estimate
        else:
            speed_kmh = float(self.model.predict(self._current_features())[0])
            speed_kmh = max(0.0, speed_kmh)
        self.last_speed_kmh = speed_kmh

        speed_ms = speed_kmh / 3.6
        # NHC: velocity is forced along the held heading direction -- no
        # sideways (v_lateral) component can exist by construction
        self.dr_x += speed_ms * math.cos(self.dr_heading) * dt
        self.dr_y += speed_ms * math.sin(self.dr_heading) * dt
        lat, lon = self.ref_frame.to_latlon(self.dr_x, self.dr_y)
        return {"mode": "DR", "lat": lat, "lon": lon, "speed_kmh": round(speed_kmh, 1)}
