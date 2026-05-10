import math

import numpy as np


def quaternion_normalize(quaternions):
    """Normalize quaternions to unit length.

    Args:
        quaternions: Array-like quaternions with real part first, as an array
            of shape `(..., 4)`.

    Returns:
        A `float64` NumPy array with shape `(..., 4)`. Near-zero norms are
        clamped to avoid division by zero.
    """
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norm = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions / np.maximum(norm, 1e-12)


def quaternion_invert(quaternion):
    """Invert unit quaternions.

    Args:
        quaternion: Array-like unit quaternions with real part first, as an
            array of shape `(..., 4)`.

    Returns:
        A `float64` NumPy array with shape `(..., 4)` containing the inverse
        quaternions. For unit quaternions this is `(w, -x, -y, -z)`.
    """
    out = np.asarray(quaternion, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def quaternion_raw_multiply(a, b):
    """Multiply quaternions.

    Args:
        a: Left-hand quaternions with real part first, as an array of shape
            `(..., 4)`.
        b: Right-hand quaternions with real part first, as an array of shape
            `(..., 4)`. Usual NumPy broadcasting rules apply before the final
            quaternion dimension.

    Returns:
        A `float64` NumPy array with shape `(..., 4)` representing `a * b`.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quaternion_to_yaw(quaternions):
    """Extract yaw angles from quaternions.

    Args:
        quaternions: Quaternions with real part first, as an array of shape
            `(..., 4)`. Inputs are normalized internally.

    Returns:
        A `float64` NumPy array with shape `(...)` containing yaw angles in
        radians, using the Z-up convention.
    """
    quaternions = quaternion_normalize(quaternions)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quaternion(yaw):
    """Convert yaw angles to Z-axis quaternions.

    Args:
        yaw: Array-like yaw angles in radians with shape `(...)`.

    Returns:
        A `float64` NumPy array with shape `(..., 4)` in `(w, x, y, z)` order.
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    half = yaw * 0.5
    return np.stack(
        [np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)],
        axis=-1,
    )


def rotate_xy_by_yaw(xy, yaw):
    """Rotate XY vectors by yaw angles.

    Args:
        xy: Array-like vectors with shape `(..., 2)`.
        yaw: Array-like yaw angles in radians with shape `(...)` or any shape
            broadcast-compatible with `xy[..., 0]`.

    Returns:
        A `float64` NumPy array with shape `(..., 2)` containing rotated XY
        vectors.
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def rotate_xy_by_inverse_yaw(xy, yaw):
    """Rotate XY vectors by the inverse of yaw angles.

    Args:
        xy: Array-like vectors with shape `(..., 2)`.
        yaw: Array-like yaw angles in radians with shape `(...)` or any shape
            broadcast-compatible with `xy[..., 0]`.

    Returns:
        A `float64` NumPy array with shape `(..., 2)` containing vectors
        rotated by `-yaw`.
    """
    return rotate_xy_by_yaw(xy, -yaw)


def quaternion_to_matrix(quaternions):
    """Convert quaternions to rotation matrices.

    Args:
        quaternions: Quaternions with real part first, as an array of shape
            `(..., 4)`. Inputs are normalized internally.

    Returns:
        A `float64` NumPy array with shape `(..., 3, 3)`.
    """
    quaternions = quaternion_normalize(quaternions)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.stack(
        [
            np.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], axis=-1),
        ],
        axis=-2,
    )


def matrix_to_quaternion(matrix):
    """Convert rotation matrices to quaternions.

    Args:
        matrix: Array-like rotation matrices with shape `(..., 3, 3)`.

    Returns:
        A normalized `float64` NumPy array with shape `(..., 4)` in
        `(w, x, y, z)` order.
    """
    m = np.asarray(matrix, dtype=np.float64)
    flat = m.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 4), dtype=np.float64)
    for i, mat in enumerate(flat):
        trace = np.trace(mat)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            out[i] = [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ]
        else:
            axis = int(np.argmax(np.diag(mat)))
            if axis == 0:
                s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
                out[i] = [
                    (mat[2, 1] - mat[1, 2]) / s,
                    0.25 * s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                ]
            elif axis == 1:
                s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
                out[i] = [
                    (mat[0, 2] - mat[2, 0]) / s,
                    (mat[0, 1] + mat[1, 0]) / s,
                    0.25 * s,
                    (mat[1, 2] + mat[2, 1]) / s,
                ]
            else:
                s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
                out[i] = [
                    (mat[1, 0] - mat[0, 1]) / s,
                    (mat[0, 2] + mat[2, 0]) / s,
                    (mat[1, 2] + mat[2, 1]) / s,
                    0.25 * s,
                ]
    return quaternion_normalize(out.reshape(m.shape[:-2] + (4,)))


def matrix_to_rotation_6d(matrix):
    """Convert rotation matrices to the continuous 6D representation.

    Args:
        matrix: Array-like rotation matrices with shape `(..., 3, 3)`.

    Returns:
        A `float64` NumPy array with shape `(..., 6)` containing the first two
        matrix rows flattened. This row-major convention matches the G1
        dataset representation used by `data_loaders.g1`.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6):
    """Convert the continuous 6D representation to rotation matrices.

    Args:
        d6: Array-like rotations with shape `(..., 6)`, interpreted as two
            row vectors in the row-major convention used by
            `matrix_to_rotation_6d`.

    Returns:
        A `float64` NumPy array with shape `(..., 3, 3)` containing orthonormal
        rotation matrices.
    """
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)
