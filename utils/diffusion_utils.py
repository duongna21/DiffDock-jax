import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from utils.geometry import rigid_transform_Kabsch_3D_jax, axis_angle_to_matrix
from utils.torsion_torch import modify_conformer_torsion_angles
from flax import linen as nn


def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma


def modify_conformer(data, tr_update, rot_update, torsion_updates):
    lig_center = jnp.mean(data['ligand'].pos, axis=0, keepdims=True)
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())
    rigid_new_pos = jnp.matmul(data['ligand'].pos - lig_center, rot_mat.T) + tr_update + lig_center

    if torsion_updates is not None:
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                               data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                               data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, jnp.ndarray) else data['ligand'].mask_rotate[0],
                                                               torsion_updates)
        R, t = rigid_transform_Kabsch_3D_jax(flexible_new_pos.T, rigid_new_pos.T)
        aligned_flexible_pos = jnp.matmul(flexible_new_pos, R.T) + t.T
        data['ligand'].pos = aligned_flexible_pos
    else:
        data['ligand'].pos = rigid_new_pos
    return data


def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = jnp.log(max_positions) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class GaussianFourierProjection(nn.Module):
    embedding_size: int = 156
    scale: float = 1.0

    def setup(self):
        self.W = self.param('W', (self.embedding_size // 2,), initializer=nn.initializers.normal()) * self.scale

    def __call__(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * jnp.pi
        emb = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        return emb

def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)
    else:
        raise NotImplemented
    return emb_func

def get_t_schedule(inference_steps):
    return jnp.linspace(1, 0, inference_steps + 1)[:-1]

def set_time(complex_graphs, t_tr, t_rot, t_tor, batchsize, all_atoms):
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * jnp.ones(complex_graphs['ligand'].num_nodes),
        'rot': t_rot * jnp.ones(complex_graphs['ligand'].num_nodes),
        'tor': t_tor * jnp.ones(complex_graphs['ligand'].num_nodes)}
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * jnp.ones(complex_graphs['receptor'].num_nodes),
        'rot': t_rot * jnp.ones(complex_graphs['receptor'].num_nodes),
        'tor': t_tor * jnp.ones(complex_graphs['receptor'].num_nodes)}
    complex_graphs.complex_t = {
        'tr': t_tr * jnp.ones(batchsize),
        'rot': t_rot * jnp.ones(batchsize),
        'tor': t_tor * jnp.ones(batchsize)}
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * jnp.ones(complex_graphs['atom'].num_nodes),
            'rot': t_rot * jnp.ones(complex_graphs['atom'].num_nodes),
            'tor': t_tor * jnp.ones(complex_graphs['atom'].num_nodes)}
