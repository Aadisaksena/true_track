"""
test_ekf.py — Unit tests for Extended Kalman Filter (app/ekf.py)
"""
import math
import numpy as np
import pytest
from app.ekf import ExtendedKalmanFilter


def test_ekf_initialization():
    ekf = ExtendedKalmanFilter(initial_x=10.0, initial_y=20.0, initial_heading=0.0)
    state = ekf.get_state()
    assert state["x"] == 10.0
    assert state["y"] == 20.0
    assert state["heading"] == 0.0
    assert state["gyro_bias"] == 0.0


def test_ekf_predict_motion():
    ekf = ExtendedKalmanFilter(initial_x=0.0, initial_y=0.0, initial_heading=0.0)
    # Accelerate forward at 1.0 m/s^2 along X axis for 1 second (10 steps of 0.1s)
    for _ in range(10):
        ekf.predict(ax_veh=1.0, ay_veh=0.0, gyro_yaw=0.0, dt=0.1)

    state = ekf.get_state()
    # v = a * t = 1.0 m/s
    assert abs(state["vx"] - 1.0) < 0.1
    assert state["x"] > 0.4  # s = 0.5 * a * t^2 ~ 0.5m


def test_ekf_gnss_update():
    ekf = ExtendedKalmanFilter(initial_x=0.0, initial_y=0.0)
    # Predict forward to drift slightly
    ekf.predict(ax_veh=0.5, ay_veh=0.0, gyro_yaw=0.0, dt=1.0)
    
    # GNSS fix arrives at (1.0, 0.0)
    ekf.update_gnss(gnss_x=1.0, gnss_y=0.0)
    state = ekf.get_state()
    assert abs(state["x"] - 1.0) < 0.5


def test_ekf_ai_motion_update():
    ekf = ExtendedKalmanFilter(initial_x=0.0, initial_y=0.0, initial_heading=0.0)
    # Update state with AI predicted speed of 10 m/s (36 km/h)
    ekf.update_ai_motion(ai_speed_ms=10.0)
    state = ekf.get_state()
    assert abs(state["speed_ms"] - 10.0) < 2.0
