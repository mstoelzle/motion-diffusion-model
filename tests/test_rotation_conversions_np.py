import numpy as np
import torch

from utils.rotation_conversions_np import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_invert,
    quaternion_raw_multiply,
    quaternion_normalize,
    quaternion_to_matrix,
    quaternion_to_yaw,
    rotate_xy_by_inverse_yaw,
    rotate_xy_by_yaw,
    rotation_6d_to_matrix,
    yaw_to_quaternion,
)
from utils.rotation_conversions import (
    matrix_to_quaternion as torch_matrix_to_quaternion,
    matrix_to_rotation_6d as torch_matrix_to_rotation_6d,
    quaternion_to_matrix as torch_quaternion_to_matrix,
    rotation_6d_to_matrix as torch_rotation_6d_to_matrix,
)


def _same_quaternion_rotation(a, b, atol=1e-6):
    a = quaternion_normalize(a)
    b = quaternion_normalize(b)
    return np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol)


def test_yaw_quaternion_roundtrip_and_shapes():
    yaw = np.array([-np.pi, -0.3, 0.0, 0.7, np.pi / 2], dtype=np.float64)
    quaternions = yaw_to_quaternion(yaw)
    recovered = quaternion_to_yaw(quaternions)

    assert quaternions.shape == (5, 4)
    assert recovered.shape == (5,)
    assert np.allclose(np.sin(recovered), np.sin(yaw), atol=1e-6)
    assert np.allclose(np.cos(recovered), np.cos(yaw), atol=1e-6)
    assert np.allclose(np.linalg.norm(quaternions, axis=-1), 1.0, atol=1e-6)


def test_quaternion_raw_multiply_invert_identity_and_composition():
    yaw_a = 0.4
    yaw_b = -1.2
    quaternion_a = yaw_to_quaternion(yaw_a)
    quaternion_b = yaw_to_quaternion(yaw_b)

    identity = quaternion_raw_multiply(quaternion_a, quaternion_invert(quaternion_a))
    composed = quaternion_raw_multiply(quaternion_a, quaternion_b)

    assert np.allclose(identity, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    assert _same_quaternion_rotation(composed, yaw_to_quaternion(yaw_a + yaw_b))


def test_rotate_xy_by_yaw_and_inverse():
    xy = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]], dtype=np.float64)
    yaw = np.array([np.pi / 2, np.pi, -np.pi / 4], dtype=np.float64)

    rotated = rotate_xy_by_yaw(xy, yaw)
    restored = rotate_xy_by_inverse_yaw(rotated, yaw)

    assert rotated.shape == xy.shape
    assert np.allclose(rotated[0], np.array([0.0, 1.0]), atol=1e-6)
    assert np.allclose(restored, xy, atol=1e-6)


def test_quaternion_matrix_roundtrip_batched():
    quaternions = quaternion_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, -0.3, 0.4],
                yaw_to_quaternion(-0.75),
            ],
            dtype=np.float64,
        )
    )
    matrix = quaternion_to_matrix(quaternions)
    recovered = matrix_to_quaternion(matrix)

    assert matrix.shape == (3, 3, 3)
    assert recovered.shape == (3, 4)
    assert np.allclose(matrix @ np.swapaxes(matrix, -1, -2), np.eye(3), atol=1e-6)
    assert np.all([_same_quaternion_rotation(a, b) for a, b in zip(quaternions, recovered)])


def test_numpy_helpers_match_torch_rotation_conversions_for_shared_apis():
    quaternions = quaternion_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.8, 0.2, -0.3, 0.4],
                yaw_to_quaternion(-0.75),
            ],
            dtype=np.float64,
        )
    )
    torch_quaternions = torch.from_numpy(quaternions)

    matrices_np = quaternion_to_matrix(quaternions)
    matrices_torch = torch_quaternion_to_matrix(torch_quaternions).numpy()
    d6_np = matrix_to_rotation_6d(matrices_np)
    d6_torch = torch_matrix_to_rotation_6d(torch.from_numpy(matrices_np)).numpy()

    assert np.allclose(matrices_np, matrices_torch, atol=1e-6)
    assert np.allclose(d6_np, d6_torch, atol=1e-6)
    assert np.allclose(rotation_6d_to_matrix(d6_np), torch_rotation_6d_to_matrix(torch.from_numpy(d6_np)).numpy(), atol=1e-6)
    assert np.all(
        [
            _same_quaternion_rotation(a, b)
            for a, b in zip(
                matrix_to_quaternion(matrices_np),
                torch_matrix_to_quaternion(torch.from_numpy(matrices_np)).numpy(),
            )
        ]
    )


def test_matrix_to_quaternion_handles_180_degree_axis_rotations():
    matrix = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    recovered = matrix_to_quaternion(matrix)

    assert recovered.shape == (3, 4)
    assert np.all([_same_quaternion_rotation(a, b) for a, b in zip(expected, recovered)])


def test_rotation_6d_matrix_roundtrip_row_major_convention():
    matrix = quaternion_to_matrix(yaw_to_quaternion(np.array([0.1, -0.9], dtype=np.float64)))
    d6 = matrix_to_rotation_6d(matrix)
    recovered = rotation_6d_to_matrix(d6)

    assert d6.shape == (2, 6)
    assert recovered.shape == (2, 3, 3)
    assert np.allclose(d6, matrix[..., :2, :].reshape(2, 6), atol=1e-6)
    assert np.allclose(recovered, matrix, atol=1e-6)
    assert np.allclose(recovered @ np.swapaxes(recovered, -1, -2), np.eye(3), atol=1e-6)


def test_quaternion_normalize_handles_near_zero_inputs():
    quaternions = quaternion_normalize(np.zeros((2, 4), dtype=np.float64))

    assert quaternions.shape == (2, 4)
    assert np.all(np.isfinite(quaternions))
    assert np.allclose(quaternions, 0.0)
