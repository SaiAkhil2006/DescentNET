"""
Physics utilities and mathematical foundations for 6-DOF rigid-body dynamics.

Coordinate Conventions:
- World Frame (W):
  * Origin (0,0,0) at the center of the lunar landing pad.
  * +Z points vertically upward (Zenith), away from the lunar surface.
  * +X points forward (North).
  * +Y points sideways (East), satisfying right-handed rule Z = X x Y.
  * Gravity: g_W = [0, 0, -1.622] m/s^2.

- Spacecraft Body Frame (B):
  * Origin at Spacecraft Center of Mass (CoM).
  * +Z_b points longitudinally along spacecraft apex (thrust fires along -Z_b, pushing along +Z_b).
  * +X_b points forward.
  * +Y_b points transverse/starboard.

- Quaternions:
  * MuJoCo convention: q = [qw, qx, qy, qz] where qw is the scalar real part.
  * Unit norm constraint: ||q|| = 1.
"""

import numpy as np

# Lunar gravitational acceleration (m/s^2)
MOON_GRAVITY = 1.622


def quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert a unit quaternion q = [w, x, y, z] to a 3x3 rotation matrix R_B^W.
    Transforms vectors from body frame to world frame: v_W = R_B^W @ v_B.
    """
    w, x, y, z = q
    # Normalize to avoid numerical drift
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm > 1e-12:
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
    else:
        return np.eye(3)

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion q = [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
    Convention: Z-Y-X (yaw-pitch-roll) intrinsic rotation.
    """
    w, x, y, z = q
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)  # Gimbal lock clamp
    else:
        pitch = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles [roll, pitch, yaw] in radians to unit quaternion [w, x, y, z].
    Convention: Z-Y-X rotation.
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Rotate vector v by unit quaternion q = [w, x, y, z].
    Transforms vector from body frame to world frame: v_world = R(q) @ v.
    """
    R = quat_to_rot_mat(q)
    return R @ v


def get_body_z_in_world(q: np.ndarray) -> np.ndarray:
    """
    Returns the body +Z axis (thrust direction) expressed in world coordinates.
    For an upright spacecraft, body +Z aligns with world +Z: [0, 0, 1].
    """
    w, x, y, z = q
    # Third column of rotation matrix R_B^W
    uz_x = 2.0 * (x * z + w * y)
    uz_y = 2.0 * (y * z - w * x)
    uz_z = 1.0 - 2.0 * (x * x + y * y)
    return np.array([uz_x, uz_y, uz_z], dtype=np.float64)


def tilt_angle_from_upright(q: np.ndarray) -> float:
    """
    Computes the tilt angle (in radians) of the spacecraft relative to the vertical world +Z axis.
    0 radians means perfectly upright.
    """
    uz = get_body_z_in_world(q)
    # Cosine of angle between body Z and world Z [0, 0, 1] is uz[2]
    cos_angle = np.clip(uz[2], -1.0, 1.0)
    return float(np.arccos(cos_angle))
