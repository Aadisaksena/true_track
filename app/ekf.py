"""
ekf.py — Extended Kalman Filter (EKF) for S.A.F.A.R. Sensor & Motion Fusion

State Vector (8x1):
  x = [
    px,       # Local East (meters)
    py,       # Local North (meters)
    vx,       # Velocity East (m/s)
    vy,       # Velocity North (m/s)
    heading,  # Yaw / Heading angle in radians from East
    bg,       # Gyroscope bias (rad/s)
    bax,      # Accelerometer X bias (m/s^2)
    bay       # Accelerometer Y bias (m/s^2)
  ]
"""

import math
import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, initial_x=0.0, initial_y=0.0, initial_vx=0.0, initial_vy=0.0, initial_heading=0.0):
        # State vector initialization (8x1)
        self.x = np.array([
            [initial_x],
            [initial_y],
            [initial_vx],
            [initial_vy],
            [initial_heading],
            [0.0],  # gyro bias
            [0.0],  # accel_x bias
            [0.0]   # accel_y bias
        ], dtype=float)

        # State Covariance Matrix (8x8)
        self.P = np.diag([
            1.0, 1.0,     # position uncertainty (m^2)
            0.5, 0.5,     # velocity uncertainty (m/s)^2
            0.05,         # heading uncertainty (rad^2)
            0.001,        # gyro bias uncertainty (rad/s)^2
            0.01, 0.01    # accel bias uncertainty (m/s^2)^2
        ])

        # Process Noise Covariance (8x8)
        self.Q = np.diag([
            0.01, 0.01,   # position process noise
            0.1, 0.1,     # velocity process noise
            0.005,        # heading process noise
            1e-6,         # gyro bias drift
            1e-5, 1e-5    # accel bias drift
        ])

    def predict(self, ax_veh, ay_veh, gyro_yaw, dt=0.1):
        """
        Kinematic motion prediction step driven by vehicle-frame accelerations and gyro rate.
        """
        px, py, vx, vy, heading, bg, bax, bay = self.x.flatten()

        # Correct sensor measurements for estimated biases
        gyro_corrected = gyro_yaw - bg
        ax_corrected = ax_veh - bax
        ay_corrected = ay_veh - bay

        # Heading update
        new_heading = heading + gyro_corrected * dt
        # Normalize heading to [-pi, pi]
        new_heading = math.atan2(math.sin(new_heading), math.cos(new_heading))

        # Rotate vehicle-frame accelerations to world frame
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ax_world = cos_h * ax_corrected - sin_h * ay_corrected
        ay_world = sin_h * ax_corrected + cos_h * ay_corrected

        # State propagation equations
        new_px = px + vx * dt + 0.5 * ax_world * (dt ** 2)
        new_py = py + vy * dt + 0.5 * ay_world * (dt ** 2)
        new_vx = vx + ax_world * dt
        new_vy = vy + ay_world * dt

        self.x = np.array([
            [new_px], [new_py], [new_vx], [new_vy],
            [new_heading], [bg], [bax], [bay]
        ], dtype=float)

        # Compute Jacobian matrix F_k = d f(x) / d x
        F = np.eye(8, dtype=float)
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Derivatives of position and velocity w.r.t heading
        d_ax_d_h = -sin_h * ax_corrected - cos_h * ay_corrected
        d_ay_d_h = cos_h * ax_corrected - sin_h * ay_corrected
        F[0, 4] = 0.5 * d_ax_d_h * (dt ** 2)
        F[1, 4] = 0.5 * d_ay_d_h * (dt ** 2)
        F[2, 4] = d_ax_d_h * dt
        F[3, 4] = d_ay_d_h * dt

        # Derivatives w.r.t gyro bias
        F[4, 5] = -dt

        # Derivatives w.r.t accel biases
        F[2, 6] = -cos_h * dt
        F[2, 7] = sin_h * dt
        F[3, 6] = -sin_h * dt
        F[3, 7] = -cos_h * dt

        # Propagate covariance P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, gnss_x, gnss_y, gnss_vx=None, gnss_vy=None, gnss_heading=None, R_gnss=None):
        """
        GNSS position, velocity & heading measurement update.
        """
        if R_gnss is None:
            R_gnss = np.diag([0.5, 0.5]) # Tight GNSS measurement covariance

        z = np.array([[gnss_x], [gnss_y]], dtype=float)
        H = np.zeros((2, 8), dtype=float)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        y = z - H @ self.x # Measurement residual
        S = H @ self.P @ H.T + R_gnss # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

        if gnss_vx is not None and gnss_vy is not None:
            z_v = np.array([[gnss_vx], [gnss_vy]], dtype=float)
            H_v = np.zeros((2, 8), dtype=float)
            H_v[0, 2] = 1.0
            H_v[1, 3] = 1.0
            R_v = np.diag([0.2, 0.2])
            y_v = z_v - H_v @ self.x
            S_v = H_v @ self.P @ H_v.T + R_v
            K_v = self.P @ H_v.T @ np.linalg.inv(S_v)
            self.x = self.x + K_v @ y_v
            self.P = (np.eye(8) - K_v @ H_v) @ self.P

        if gnss_heading is not None:
            y_head = math.atan2(math.sin(gnss_heading - self.x[4, 0]), math.cos(gnss_heading - self.x[4, 0]))
            H_head = np.zeros((1, 8), dtype=float)
            H_head[0, 4] = 1.0
            R_head = 0.02
            S_head = H_head @ self.P @ H_head.T + R_head
            K_head = self.P @ H_head.T / S_head[0, 0]
            self.x = self.x + K_head * y_head
            self.P = (np.eye(8) - K_head @ H_head) @ self.P

    def update_ai_motion(self, ai_speed_ms, mag_heading=None, R_speed=0.05, R_heading=0.04):
        """
        AI Motion & NHC constraint update step during GNSS outage.
        Measurement model:
          - Vehicle velocity magnitude matches AI speed prediction
          - Sideways velocity component is 0 (Non-Holonomic Constraint)
          - Magnetometer heading (optional)
        """
        heading = self.x[4, 0]
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        # 1. Forward Speed & Zero Sideways Speed Updates
        # Forward speed: v_forward = vx * cos(h) + vy * sin(h)
        # Sideways speed: v_side = -vx * sin(h) + vy * cos(h) = 0
        z = np.array([[ai_speed_ms], [0.0]], dtype=float)

        vx = self.x[2, 0]
        vy = self.x[3, 0]
        v_forward = vx * cos_h + vy * sin_h
        v_side = -vx * sin_h + vy * cos_h
        hx = np.array([[v_forward], [v_side]], dtype=float)

        H = np.zeros((2, 8), dtype=float)
        # d v_forward / d vx, vy, heading
        H[0, 2] = cos_h
        H[0, 3] = sin_h
        H[0, 4] = 0.0  # Speed magnitude derivative w.r.t heading is zero

        # d v_side / d vx, vy, heading
        H[1, 2] = -sin_h
        H[1, 3] = cos_h
        H[1, 4] = 0.0

        R = np.diag([R_speed, 0.04]) # R_speed and tight lateral constraint
        y = z - hx
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ H) @ self.P

        # 2. Magnetometer Heading Update (if available and valid)
        if mag_heading is not None:
            z_head = mag_heading
            y_head = z_head - self.x[4, 0]
            # Normalize angle difference
            y_head = math.atan2(math.sin(y_head), math.cos(y_head))
            
            H_head = np.zeros((1, 8), dtype=float)
            H_head[0, 4] = 1.0

            S_head = H_head @ self.P @ H_head.T + R_heading
            K_head = self.P @ H_head.T / S_head[0, 0]

            self.x = self.x + K_head * y_head
            self.P = (np.eye(8) - K_head @ H_head) @ self.P

    def get_state(self):
        """Returns dict of current state estimates and variance confidence metrics."""
        px, py, vx, vy, heading, bg, bax, bay = self.x.flatten()
        pos_var = float(self.P[0, 0] + self.P[1, 1])
        head_var = float(self.P[4, 4])
        speed = float(math.hypot(vx, vy))

        return {
            "x": px,
            "y": py,
            "vx": vx,
            "vy": vy,
            "speed_ms": speed,
            "speed_kmh": speed * 3.6,
            "heading": heading,
            "gyro_bias": bg,
            "accel_bias_x": bax,
            "accel_bias_y": bay,
            "pos_uncertainty_m": math.sqrt(max(0.0, pos_var)),
            "heading_uncertainty_deg": math.degrees(math.sqrt(max(0.0, head_var)))
        }
