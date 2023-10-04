import jax
import jax.numpy as jnp
import numpy as np  # using this ONLY for np.ndarray type check
from scipy.spatial.transform import Rotation as R


def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    # The deep copy isn't necessary as JAX arrays are immutable
    if type(pos) != np.ndarray:
        pos = np.array(pos)

    for idx_edge, e in enumerate(edge_index):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / jnp.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos_masked = pos[mask_rotate[idx_edge]]
        pos[mask_rotate[idx_edge]] = (pos_masked - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy:
        pos = jnp.array(pos.astype(np.float32))  # sends the data to the device
    return pos
