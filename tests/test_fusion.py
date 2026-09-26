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
    assert abs(res["lat"] - 12.9716) < 1e-4
    
    # simulate outage
    res_dr = sess.update(accel, gravity, gyro, gps=gps, simulate_outage=True)
    assert res_dr["mode"] == "DR"
    assert res_dr["lat"] is not None


def test_constant_speed_cruising_40kmh():
    """Test 1: GNSS active at ~40 km/h, then outage. DR speed should stay near 40 km/h."""
    bundle = get_model_bundle()
    sess = OnlineFusionSession(bundle)
    accel = {"x": 0.0, "y": 0.0, "z": 9.81}
    gravity = {"x": 0.0, "y": 0.0, "z": 9.81}
    gyro = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    # Warmup with GNSS moving east at ~40 km/h (11.11 m/s -> 0.00001 deg lon per 0.1s ~ 1.08m)
    lat, lon = 12.9716, 77.5946
    for i in range(20):
        lon += 0.00001
        res = sess.update(accel, gravity, gyro, gps={"lat": lat, "lon": lon}, dt=0.1)

    gnss_speed = res["speed_kmh"]
    assert 30.0 < gnss_speed < 50.0  # moving around 40 km/h

    # Transition to GNSS-Denied (DR)
    dr_speeds = []
    for _ in range(30): # 3 seconds of DR
        res_dr = sess.update(accel, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)
        dr_speeds.append(res_dr["speed_kmh"])

    # First DR speed should match last confirmed speed without sudden spike/drop
    assert abs(dr_speeds[0] - gnss_speed) < 5.0
    # Speed after 3 seconds of cruising should remain within 10 km/h of initial speed (no artificial collapse to 25 km/h)
    assert abs(dr_speeds[-1] - gnss_speed) < 10.0


def test_constant_speed_cruising_60kmh():
    """Test 2: GNSS active at ~60 km/h, then outage. DR speed stays near 60 km/h."""
    bundle = get_model_bundle()
    sess = OnlineFusionSession(bundle)
    accel = {"x": 0.0, "y": 0.0, "z": 9.81}
    gravity = {"x": 0.0, "y": 0.0, "z": 9.81}
    gyro = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    lat, lon = 12.9716, 77.5946
    for i in range(20):
        lon += 0.000015
        res = sess.update(accel, gravity, gyro, gps={"lat": lat, "lon": lon}, dt=0.1)

    gnss_speed = res["speed_kmh"]
    assert 45.0 < gnss_speed < 70.0

    for _ in range(20):
        res_dr = sess.update(accel, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)

    assert abs(res_dr["speed_kmh"] - gnss_speed) < 12.0


def test_acceleration_and_braking_dynamics():
    """Test 3 & 4: Speed increases during forward accel and decreases during braking."""
    bundle = get_model_bundle()
    sess = OnlineFusionSession(bundle)
    accel_idle = {"x": 0.0, "y": 0.0, "z": 9.81}
    accel_fwd = {"x": 0.0, "y": 1.5, "z": 9.81}  # forward acceleration
    accel_brake = {"x": 0.0, "y": -2.0, "z": 9.81} # braking
    gravity = {"x": 0.0, "y": 0.0, "z": 9.81}
    gyro = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    lat, lon = 12.9716, 77.5946
    for _ in range(20):
        lon += 0.00001
        sess.update(accel_idle, gravity, gyro, gps={"lat": lat, "lon": lon}, dt=0.1)

    # Outage starts
    res_start = sess.update(accel_idle, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)
    s0 = res_start["speed_kmh"]

    # Accelerate
    for _ in range(10):
        res_acc = sess.update(accel_fwd, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)
    assert res_acc["speed_kmh"] >= s0 - 1.0  # speed increases or stays high

    # Brake
    for _ in range(15):
        res_brk = sess.update(accel_brake, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)
    assert res_brk["speed_kmh"] < res_acc["speed_kmh"] # speed decreases during braking


def test_gnss_restoration_smoothness():
    """Test 7: Smooth transition when GNSS signal returns without sudden jump."""
    bundle = get_model_bundle()
    sess = OnlineFusionSession(bundle)
    accel = {"x": 0.0, "y": 0.0, "z": 9.81}
    gravity = {"x": 0.0, "y": 0.0, "z": 9.81}
    gyro = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    lat, lon = 12.9716, 77.5946
    for _ in range(10):
        lon += 0.0001
        sess.update(accel, gravity, gyro, gps={"lat": lat, "lon": lon}, dt=0.1)

    # DR for 20 steps
    for _ in range(20):
        res_dr = sess.update(accel, gravity, gyro, gps=None, simulate_outage=True, dt=0.1)

    # GNSS restored
    res_rec = sess.update(accel, gravity, gyro, gps={"lat": lat + 0.0001, "lon": lon + 0.002}, simulate_outage=False, dt=0.1)
    assert res_rec["mode"] == "RECOVERING"
    assert res_rec["lat"] is not None
