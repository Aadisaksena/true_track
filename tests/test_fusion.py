import pytest
from app.geo import LocalFrame
from app.model import get_model_bundle
from app.fusion import OnlineFusionSession

def test_local_frame():
    ref = LocalFrame(12.9716, 77.5946)
    x, y = ref.to_xy(12.9716, 77.5946)
    assert abs(x) < 1e-4 and abs(y) < 1e-4
    lat, lon = ref.to_latlon(0, 0)
    assert abs(lat - 12.9716) < 1e-4 and abs(lon - 77.5946) < 1e-4

def test_model_bundle_loading():
    bundle = get_model_bundle()
    assert "model" in bundle
    assert "window" in bundle

def test_online_fusion_session():
    bundle = get_model_bundle()
    sess = OnlineFusionSession(bundle)
    accel = {"x": 0.0, "y": 0.0, "z": 9.81}
    gravity = {"x": 0.0, "y": 0.0, "z": 9.81}
    gyro = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    gps = {"lat": 12.9716, "lon": 77.5946}
    
    res = sess.update(accel, gravity, gyro, gps=gps)
    assert res["mode"] == "GNSS"
    assert res["lat"] == 12.9716
    
    # simulate outage
    res_dr = sess.update(accel, gravity, gyro, gps=gps, simulate_outage=True)
    assert res_dr["mode"] == "DR"
    assert res_dr["lat"] is not None
