"""
train_model.py
Trains the AI velocity model (the "virtual speedometer") on real IO-VNBD trips
and saves it as model.joblib for the backend to load.

Run: python train_model.py
"""
import sys
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))
from app.data_utils import load_trip, windowed_features, FEATURE_NAMES

HERE = Path(__file__).parent
SAMPLES = HERE / "training_data"

TRAIN_TRIPS = [
    (SAMPLES / "S1/S1/S-S1.csv", SAMPLES / "S1/S1/V-S1.csv"),
    (SAMPLES / "M/M (Driver B)/S-M.csv", SAMPLES / "M/M (Driver B)/V-M.csv"),
]
TEST_TRIP = (
    SAMPLES / "Vta/Vta (Driver E)/Vta01a/S-Vta1a.csv",
    SAMPLES / "Vta/Vta (Driver E)/Vta01a/V-Vta1a.csv",
)

WINDOW = 10  # 1 second @ 10Hz


def build_dataset(trip_paths):
    X_all, y_all = [], []
    for s_path, v_path in trip_paths:
        df = load_trip(str(s_path), str(v_path))
        X, y, _ = windowed_features(df, window=WINDOW)
        X_all.append(X)
        y_all.append(y)
    return np.vstack(X_all), np.concatenate(y_all)


def main():
    print("Loading training trips (S1, M)...")
    X_train, y_train = build_dataset(TRAIN_TRIPS)
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    print("Loading held-out test trip (Vta01a, never seen during training)...")
    X_test, y_test = build_dataset([TEST_TRIP])
    print(f"Test set: {X_test.shape[0]} samples")

    # HistGradientBoostingRegressor: as accurate as RandomForest here but orders of
    # magnitude faster to train on this dataset size -- important since the final
    # deployed model also needs to stay small/fast enough for on-device inference.
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=8, learning_rate=0.08,
        l2_regularization=0.1, random_state=42,
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, pred_train)
    mae_test = mean_absolute_error(y_test, pred_test)

    print(f"\nTrain MAE: {mae_train:.2f} km/h")
    print(f"Held-out (Vta01a) MAE: {mae_test:.2f} km/h  <-- this is the honest generalization number")

    out_path = HERE / "app" / "trained_model.joblib"
    joblib.dump({"model": model, "window": WINDOW, "feature_names": FEATURE_NAMES}, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    main()
