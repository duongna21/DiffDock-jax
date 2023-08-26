import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from utils.geometry import rigid_transform_Kabsch_3D_jax
from utils.torsion import modify_conformer_torsion_angles


def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates):
    lig_center = jnp.mean(data['ligand'].pos, axis=0, keepdims=True)
    rot_mat = axis_angle_to_matrix_jax(rot_update.squeeze())
    rigid_new_pos = jnp.matmul(data['ligand'].pos - lig_center, rot_mat.T) + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles_jax(rigid_new_pos,
                                                               data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                               data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, jnp.ndarray) else data['ligand'].mask_rotate[0],
                                                               torsion_updates)
        R, t = rigid_transform_Kabsch_3D_jax(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = jnp.matmul(flexible_new_pos, R.T) + t.T
        data['ligand'].pos = aligned_flexible_pos
    else:
        data['ligand'].pos = rigid_new_pos
    return data