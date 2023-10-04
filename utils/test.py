import torch
import numpy as np
import jax.numpy as jnp
from utils.torsion_torch import modify_conformer_torsion_angles_jax

def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()

    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]
        assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

def test_modify_conformer_torsion_angles_torch():
    pos = torch.randn((10, 3))
    edge_index = torch.tensor([[1, 2], [3, 4]])
    mask_rotate = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    torsion_updates = torch.tensor([0.5, -0.5])

    out_torch = modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates)
    return out_torch

def test_modify_conformer_torsion_angles_jax():
    pos = jnp.array(torch.randn((10, 3)))
    edge_index = jnp.array([[1, 2], [3, 4]])
    mask_rotate = jnp.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    torsion_updates = jnp.array([0.5, -0.5])

    out_jax = modify_conformer_torsion_angles_jax(pos, edge_index, mask_rotate, torsion_updates)
    return out_jax

def test_comparison():
    torch_result = test_modify_conformer_torsion_angles_torch().numpy()
    jax_result = np.array(test_modify_conformer_torsion_angles_jax())

    assert np.allclose(torch_result, jax_result, atol=1e-6), "Results don't match!"

test_comparison()