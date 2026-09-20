"""
data_utils.py
Loads IO-VNBD Synchronised S-*/V-* trip pairs and turns them into
windowed feature vectors for the AI velocity model.

Why these specific columns:
- S-file (smartphone): ACCELEROMETER X/Y/Z, GRAVITY X/Y/Z, GYROSCOPE Yaw/Pitch/Roll
  are exactly what a real phone mounted on a dashboard can give us live.
- V-file (vehicle CAN bus): 'Indicated Vehicle Speed (km/hr)' is the REAL speedometer
  reading -> this is the ground-truth label our AI model is trained to reproduce
  WITHOUT ever seeing the OBD-II feed at inference time.
- Both files are already row-aligned (verified: identical row counts, 10Hz), so no
  timestamp-matching step is needed for this dataset.
"""
import numpy as np
import pandas as pd

# IO-VNBD's S-*.csv files mix UTF-8 and Latin-1 encoded special characters
# (degree signs, micro signs, superscripts) within the SAME file, so no single
# encoding reads every column cleanly. We sidestep this by matching columns on
# their stable ASCII prefix rather than the exact (encoding-fragile) name.
ACCEL_PREFIXES = ["ACCELEROMETER X", "ACCELEROMETER Y", "ACCELEROMETER Z"]
GRAV_PREFIXES = ["GRAVITY X", "GRAVITY Y", "GRAVITY Z"]
GYRO_PREFIXES = ["GYROSCOPE Yaw", "GYROSCOPE Pitch", "GYROSCOPE Roll"]
ORIENT_PREFIXES = ["ORIENTATION (Yaw)", "ORIENTATION (Pitch)", "ORIENTATION (Roll"]

SPEED_LABEL_COL = "Indicated Vehicle Speed (km/hr)"
GPS_LAT_COL = "Latitude (degrees)"
GPS_LON_COL = "Longitude (degrees)"


def _find_col(columns, prefix: str) -> str:
    for c in columns:
        if c.startswith(prefix):
            return c
    raise KeyError(f"No column starting with {prefix!r} found in {list(columns)}")


def load_trip(s_path: str, v_path: str) -> pd.DataFrame:
    """Load one S/V pair and merge into a single per-sample DataFrame."""
    s = pd.read_csv(s_path, encoding="latin1")
    v = pd.read_csv(v_path, encoding="latin1")
    s.columns = [c.strip() for c in s.columns]
    v.columns = [c.strip() for c in v.columns]

    n = min(len(s), len(v))
    s, v = s.iloc[:n].reset_index(drop=True), v.iloc[:n].reset_index(drop=True)

    out_names = [
        "accel_x", "accel_y", "accel_z",
        "gravity_x", "gravity_y", "gravity_z",
        "gyro_yaw", "gyro_pitch", "gyro_roll",
        "orient_yaw", "orient_pitch", "orient_roll",
    ]
    prefixes = ACCEL_PREFIXES + GRAV_PREFIXES + GYRO_PREFIXES + ORIENT_PREFIXES

    df = pd.DataFrame()
    for out_name, prefix in zip(out_names, prefixes):
        df[out_name] = s[_find_col(s.columns, prefix)]

    df["speed_kmh"] = v[SPEED_LABEL_COL]
    df["lat"] = v[GPS_LAT_COL]
    df["lon"] = v[GPS_LON_COL]
    df["dt"] = 0.1  # confirmed 10Hz in both files
    df["t"] = df["dt"].cumsum()
    return df


def windowed_features(df: pd.DataFrame, window: int = 10, long_window: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build one feature row per timestep from two window scales:
    - `window` (default 1s @ 10Hz): short-term dynamics -- braking/turning shocks.
    - `long_window` (default 3s): road/engine VIBRATION ENERGY, which is what
      actually correlates with speed. Instantaneous acceleration barely correlates
      with speed at all (measured corr ~0.02-0.22 on real trips) because
      acceleration is speed's derivative -- a car cruising at 100km/h and a parked
      car both have near-zero net acceleration. What DOES scale with speed is
      vibration intensity (rolling std of vertical accel, corr ~0.62) and steering
      micro-corrections (rolling mean |gyro yaw|, corr ~0.67) -- both measured on
      the real IO-VNBD S1 trip. This is the actual signal the AI model learns from.
    """
    lin_x = df["accel_x"] - df["gravity_x"]
    lin_y = df["accel_y"] - df["gravity_y"]
    lin_z = df["accel_z"] - df["gravity_z"]
    accel_mag = np.sqrt(lin_x ** 2 + lin_y ** 2 + lin_z ** 2)
    gyaw, gpitch, groll = df["gyro_yaw"], df["gyro_pitch"], df["gyro_roll"]

    r = lambda s, w=window: s.rolling(w)  # noqa: E731
    rl = lambda s: s.rolling(long_window)  # noqa: E731
    feat_df = pd.DataFrame({
        "lin_x_std": r(lin_x).std(),
        "lin_y_std": r(lin_y).std(),
        "lin_z_std": r(lin_z).std(),
        "accel_mag_mean": r(accel_mag).mean(), "accel_mag_max": r(accel_mag).max(),
        "gyro_yaw_std": r(gyaw).std(),
        "gyro_pitch_mean": r(gpitch).mean(), "gyro_roll_mean": r(groll).mean(),
        # long-window vibration-energy features (the strongest speed predictors)
        "lin_z_std_3s": rl(lin_z).std(),
        "accel_mag_std_3s": rl(accel_mag).std(),
        "gyro_yaw_absmean_3s": rl(gyaw.abs()).mean(),
        "gyro_yaw_std_3s": rl(gyaw).std(),
    })

    valid = feat_df.notna().all(axis=1)
    valid.iloc[:long_window] = False  # rolling() leaves partial windows at the start
    idxs = np.where(valid.values)[0]

    X = feat_df.loc[valid, FEATURE_NAMES_INTERNAL].values
    y = df["speed_kmh"].values[idxs]
    return X, y, idxs


def compute_features_at(df: pd.DataFrame, i: int, window: int = 10, long_window: int = 30) -> np.ndarray:
    """
    Single-sample version of windowed_features, for online/live use where we
    don't have (and don't want) the whole trip's rolling series precomputed --
    e.g. one sample just arrived from a live phone or a replay stream.
    Uses the SAME feature definitions as windowed_features (kept manually in
    sync -- see FEATURE_NAMES_INTERNAL for the canonical list/order).
    Caller must ensure i >= long_window (enough history buffered).
    """
    lo_short, lo_long = i - window + 1, i - long_window + 1
    lin_x = (df["accel_x"] - df["gravity_x"]).values
    lin_y = (df["accel_y"] - df["gravity_y"]).values
    lin_z = (df["accel_z"] - df["gravity_z"]).values
    accel_mag = np.sqrt(lin_x ** 2 + lin_y ** 2 + lin_z ** 2)
    gyaw = df["gyro_yaw"].values
    gpitch = df["gyro_pitch"].values
    groll = df["gyro_roll"].values

    sl_s = slice(lo_short, i + 1)
    sl_l = slice(lo_long, i + 1)
    return np.array([
        lin_x[sl_s].std(), lin_y[sl_s].std(), lin_z[sl_s].std(),
        accel_mag[sl_s].mean(), accel_mag[sl_s].max(),
        gyaw[sl_s].std(), gpitch[sl_s].mean(), groll[sl_s].mean(),
        lin_z[sl_l].std(), accel_mag[sl_l].std(),
        np.abs(gyaw[sl_l]).mean(), gyaw[sl_l].std(),
    ])


FEATURE_NAMES_INTERNAL = [
    "lin_x_std", "lin_y_std", "lin_z_std",
    "accel_mag_mean", "accel_mag_max",
    "gyro_yaw_std", "gyro_pitch_mean", "gyro_roll_mean",
    "lin_z_std_3s", "accel_mag_std_3s", "gyro_yaw_absmean_3s", "gyro_yaw_std_3s",
]
FEATURE_NAMES = FEATURE_NAMES_INTERNAL
