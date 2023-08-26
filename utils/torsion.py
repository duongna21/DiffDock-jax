import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R

def modify_conformer_torsion_angles_jax(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = jnp.array(pos)

    for idx_edge, e in enumerate(edge_index):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / jnp.linalg.norm(rot_vec)
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos = jax.ops.index_update(
            pos,
            jax.ops.index[mask_rotate[idx_edge]],
            (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
        )

    if not as_numpy:
        pos = jnp.array(pos, dtype=jnp.float32)
    return pos
