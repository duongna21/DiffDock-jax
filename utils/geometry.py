import jax
import jax.numpy as jnp
import math


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = jnp.split(quaternions, 4, axis=-1)
    two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)

    o = jnp.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    # return o.reshape(quaternions.shape[:-1] + (-1, 3, 3))
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = jnp.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = jnp.abs(angles) < eps
    sin_half_angles_over_angles = jnp.where(
        small_angles,
        0.5 - (angles * angles) / 48,
        jnp.sin(half_angles) / angles,
    )
    # sin_half_angles_over_angles[~small_angles] = (
    #         torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # )
    # # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # # so sin(x/2)/x is about 1/2 - (x*x)/48
    # sin_half_angles_over_angles[small_angles] = (
    #         0.5 - (angles[small_angles] * angles[small_angles]) / 48
    # )
    
    quaternions = jnp.concatenate(
        [jnp.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rigid_transform_Kabsch_3D_jax(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    centroid_A = jnp.mean(A, axis=1, keepdims=True)
    centroid_B = jnp.mean(B, axis=1, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    U, S, Vt = jnp.linalg.svd(H)

    R = Vt.T @ U.T
    if jnp.linalg.det(R) < 0:
        SS = jnp.diag(jnp.array([1., 1., -1.]))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(jnp.linalg.det(R) - 1) < 3e-3 

    t = -R @ centroid_A + centroid_B
    return R, t
