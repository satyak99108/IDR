"""
IDR MVP -- INS Math Utilities
==============================
Quaternion mathematics and geodetic helper functions for the strapdown INS.

All quaternions use the scalar-first convention: q = [w, x, y, z].
All angles are in radians unless otherwise specified.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion Operations
# ---------------------------------------------------------------------------

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Returns a unit quaternion.

    Args:
        q: 4-element array [w, x, y, z].

    Returns:
        Normalized quaternion. Returns identity [1,0,0,0] if norm is zero.
    """
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.

    Args:
        q1: [w1, x1, y1, z1]
        q2: [w2, x2, y2, z2]

    Returns:
        Product quaternion [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """
    Converts ZYX Euler angles (yaw, pitch, roll) to quaternion [w, x, y, z].

    Convention: Vehicle NED frame.
      - Roll  φ  : rotation around X (forward) axis
      - Pitch θ  : rotation around Y (right) axis
      - Yaw   ψ  : rotation around Z (down) axis

    Args:
        roll_rad:  Roll angle in radians.
        pitch_rad: Pitch angle in radians.
        yaw_rad:   Yaw (heading) angle in radians.

    Returns:
        Unit quaternion [w, x, y, z].
    """
    cr = np.cos(roll_rad * 0.5)
    sr = np.sin(roll_rad * 0.5)
    cp = np.cos(pitch_rad * 0.5)
    sp = np.sin(pitch_rad * 0.5)
    cy = np.cos(yaw_rad * 0.5)
    sy = np.sin(yaw_rad * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return normalize_quaternion(np.array([w, x, y, z]))


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Converts a unit quaternion [w, x, y, z] to a 3x3 rotation matrix.

    The matrix R satisfies: v_world = R @ v_body.

    Args:
        q: Unit quaternion [w, x, y, z].

    Returns:
        3x3 rotation matrix.
    """
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quaternion_to_euler(q: np.ndarray):
    """
    Converts quaternion [w, x, y, z] to ZYX Euler angles (roll, pitch, yaw) in radians.

    Returns:
        Tuple (roll_rad, pitch_rad, yaw_rad).
    """
    q = normalize_quaternion(q)
    w, x, y, z = q

    # Roll (φ)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) — clamped to avoid singularity
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def angular_velocity_to_quaternion_rate(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """
    Computes the quaternion derivative given the angular velocity vector.

    dq/dt = 0.5 * Omega(omega) * q

    where Omega(ω) is the skew-symmetric omega matrix in quaternion form.

    Args:
        q:     Current quaternion [w, x, y, z].
        omega: Angular velocity vector [wx, wy, wz] in rad/s (body frame).

    Returns:
        Quaternion rate dq/dt as [dw, dx, dy, dz].
    """
    wx, wy, wz = omega
    omega_matrix = np.array([
        [ 0.0, -wx, -wy, -wz],
        [ wx,  0.0,  wz, -wy],
        [ wy, -wz,  0.0,  wx],
        [ wz,  wy,  -wx, 0.0],
    ])
    return 0.5 * (omega_matrix @ q)


# ---------------------------------------------------------------------------
# Geodetic / NED Helpers
# ---------------------------------------------------------------------------

# WGS-84 constants
_WGS84_A = 6_378_137.0          # Semi-major axis (m)
_WGS84_E2 = 6.694_379_990_14e-3  # First eccentricity squared


def earth_radii(lat_rad: float):
    """
    Returns the Earth's meridian (N) and transverse (M) radii at a given latitude.

    Args:
        lat_rad: Geodetic latitude in radians.

    Returns:
        Tuple (N, M) in metres.
    """
    sin_lat = np.sin(lat_rad)
    denom = np.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    N = _WGS84_A / denom                       # Prime vertical radius
    M = _WGS84_A * (1.0 - _WGS84_E2) / denom**3  # Meridian radius
    return N, M


def latlon_update_ned(
    lat_rad: float,
    lon_rad: float,
    v_north: float,
    v_east: float,
    dt: float,
) -> tuple:
    """
    Updates geodetic latitude and longitude from NED velocity using WGS-84 radii.

    d(lat)/dt = v_north / M
    d(lon)/dt = v_east  / (N * cos(lat))

    Args:
        lat_rad:  Current latitude in radians.
        lon_rad:  Current longitude in radians.
        v_north:  North velocity component in m/s.
        v_east:   East velocity component in m/s.
        dt:       Time step in seconds.

    Returns:
        Tuple (new_lat_rad, new_lon_rad).
    """
    N, M = earth_radii(lat_rad)
    dlat = (v_north / M) * dt
    cos_lat = np.cos(lat_rad)
    if abs(cos_lat) < 1e-10:
        dlon = 0.0
    else:
        dlon = (v_east / (N * cos_lat)) * dt
    return lat_rad + dlat, lon_rad + dlon


def haversine_distance(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    """
    Returns the great-circle distance between two points in metres using the Haversine formula.
    Supports both scalar floats and numpy arrays.
    """
    R = 6_371_000.0  # Mean Earth radius (m)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_distance_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    Returns great-circle distance in metres for coordinates given in degrees.
    Supports both scalar floats and numpy arrays.
    """
    return haversine_distance(
        np.radians(lat1_deg), np.radians(lon1_deg),
        np.radians(lat2_deg), np.radians(lon2_deg),
    )


