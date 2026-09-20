"""
model.py — lazy loader for the trained AI velocity model.

Loading is deliberately lazy (not at import time) so the web process can bind
to $PORT immediately on Render; the ~750KB model is only deserialized the
first time an endpoint actually needs a speed prediction.
"""
import threading
from pathlib import Path

import joblib

_MODEL_PATH = Path(__file__).parent / "trained_model.joblib"
_bundle = None
_lock = threading.Lock()

REQUIRED_KEYS = ("model", "window", "feature_names")


def get_model_bundle():
    """Return the {model, window, feature_names} bundle, loading it once."""
    global _bundle
    if _bundle is not None:
        return _bundle

    with _lock:
        if _bundle is not None:  # another thread won the race
            return _bundle

        if not _MODEL_PATH.exists():
            raise RuntimeError(
                f"Trained model not found at {_MODEL_PATH}. It must be committed to the "
                "repository -- run `python train_model.py` locally and commit "
                "app/trained_model.joblib."
            )

        try:
            bundle = joblib.load(_MODEL_PATH)
        except ModuleNotFoundError as exc:
            # The classic cause: numpy<2 trying to read a numpy 2.x pickle.
            raise RuntimeError(
                f"Could not unpickle {_MODEL_PATH.name} ({exc}). This almost always means "
                "the installed numpy/scikit-learn versions are older than the ones the "
                "model was trained with. Install exactly what requirements.txt pins "
                "(numpy>=2.1, scikit-learn>=1.8)."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load {_MODEL_PATH.name}: {exc}") from exc

        missing = [k for k in REQUIRED_KEYS if k not in bundle]
        if missing:
            raise RuntimeError(
                f"Model bundle at {_MODEL_PATH.name} is missing key(s): {missing}"
            )

        _bundle = bundle
        return _bundle
