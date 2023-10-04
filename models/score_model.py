from typing import Optional, Any, Callable
import e3nn_jax as e3nn
import jax
import flax.linen as nn
from utils.radius import radius, radius_graph
from utils.scatter import scatter
import numpy as np
import jax.numpy as jnp
from models.batchnorm_flax import BatchNorm
from utils import torus, so3
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims
from models.linear import FunctionalLinear

class GaussianSmearing(nn.Module):
    start: float = 0.0
    stop: float = 5.0
    num_gaussians: int = 50

    def setup(self):
        self.offset = jnp.linspace(self.start, self.stop, self.num_gaussians)
        self.coeff = -0.5 / (self.offset[1] - self.offset[0]) ** 2

    def __call__(self, dist):
        dist = dist.reshape((-1, 1)) - self.offset.reshape((1, -1))
        return jnp.exp(self.coeff * jnp.power(dist, 2))

class MLP(nn.Module):

  hidden_size: int = 64
  output_size: int = 64
  dropout: float = 0.0
  bias: bool = True

  @nn.compact
  def __call__(self, inputs, training):
    x = nn.Dense(self.hidden_size, use_bias=self.bias)(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
    x = nn.Dense(self.output_size, use_bias=self.bias)(x)
    return x

class AtomEncoder(nn.Module):
    emb_dim: int = 48
    feature_dims: tuple = ()
    sigma_embed_dim: int = 32
    lm_embedding_type: str = None

    def setup(self):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        self.atom_embedding_list = []
        self.num_categorical_features = len(self.feature_dims[0])
        self.num_scalar_features = self.feature_dims[1] + self.sigma_embed_dim
        self.atom_embedding_list = [nn.Embed(num_embeddings=dim, features=self.emb_dim, embedding_init=jax.nn.initializers.glorot_uniform()) for _, dim in enumerate(self.feature_dims[0])]
        if self.num_scalar_features > 0:
            self.linear = nn.Dense(self.emb_dim)

        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else:
                raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = nn.Dense(self.emb_dim)

    def __call__(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features

        for i, emb in enumerate(self.atom_embedding_list):
            x_embedding += emb(x[:, i].astype(jnp.int32))

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(jnp.concatenate((x_embedding, x[:, -self.lm_embedding_dim:]), axis=1))
        return x_embedding


class TensorProductConvLayer(nn.Module):
    in_irreps: e3nn.Irreps
    sh_irreps: e3nn.Irreps
    out_irreps: e3nn.Irreps
    n_edge_features: int = 64
    residual: bool = True
    batch_norm: bool = True
    dropout: float = 0.0
    hidden_features: int = None

    def setup(self):
        if self.hidden_features is None:
            hidden_size = self.n_edge_features

        self.tp = tp = FunctionalLinear(e3nn.tensor_product(self.in_irreps, self.sh_irreps), self.out_irreps)
        self.fc = MLP(self.hidden_features if self.hidden_features is not None else hidden_size, tp.num_weights())
        self.batchNorm = BatchNorm(self.out_irreps) if self.batch_norm else None

    def __call__(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', training=False):

        edge_src, edge_dst = edge_index

        dense = self.fc(edge_attr, training)
        node_attr_edge_dst = e3nn.IrrepsArray(self.in_irreps, node_attr[edge_dst])
        tensor_prod = e3nn.tensor_product(node_attr_edge_dst, edge_sh)

        tp = self.tp(dense, tensor_prod)

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp.array, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = jnp.pad(node_attr, ((0,0),(0, out.shape[-1] - node_attr.shape[-1])), mode='constant')
            out = out + padded

        out = e3nn.IrrepsArray(tp.irreps, out)
        if self.batchNorm:
            out = self.batchNorm(out)
        return out


class DiffDock(nn.Module):
    t_to_sigma: Optional[Callable] = None
    timestep_emb_func: Optional[Callable] = None
    in_lig_edge_features: int = 4
    sigma_embed_dim: int = 32
    sh_lmax: int = 2
    ns: int = 16
    nv: int = 4
    num_conv_layers: int = 2
    lig_max_radius: int = 5
    rec_max_radius: int = 30
    cross_max_distance: int = 250
    center_max_distance: int = 30
    distance_embed_dim: int = 32
    cross_distance_embed_dim: int = 32
    no_torsion: bool = False
    scale_by_sigma: bool = True
    use_second_order_repr: bool = False
    batch_norm: bool = True
    dynamic_max_cross: bool = False
    dropout: float = 0.0
    lm_embedding_type: str = None

    def setup(self):
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(lmax=self.sh_lmax)
        self.lig_node_embedding = AtomEncoder(emb_dim=self.ns, feature_dims=lig_feature_dims, sigma_embed_dim=self.sigma_embed_dim)
        self.lig_edge_embedding = MLP(self.ns, self.ns, self.dropout)

        self.rec_node_embedding = AtomEncoder(emb_dim=self.ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=self.sigma_embed_dim, lm_embedding_type=self.lm_embedding_type)
        self.rec_edge_embedding = MLP(self.ns, self.ns, self.dropout)
        self.cross_edge_embedding = MLP(self.ns, self.ns, self.dropout)
        self.lig_distance_expansion = GaussianSmearing(0.0, self.lig_max_radius, self.distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, self.rec_max_radius, self.distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, self.cross_max_distance, self.cross_distance_embed_dim)

        if self.use_second_order_repr:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x2e + {self.nv}x1e + {self.nv}x2o + {self.ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{self.ns}x0e',
                f'{self.ns}x0e + {self.nv}x1o',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e',
                f'{self.ns}x0e + {self.nv}x1o + {self.nv}x1e + {self.ns}x0o'
            ]

        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        for i in range(self.num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * self.ns,
                'hidden_features': 3 * self.ns,
                'residual': False,
                'batch_norm': self.batch_norm,
                'dropout': self.dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = TensorProductConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = TensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)

        self.lig_conv_layers = lig_conv_layers
        self.rec_conv_layers = rec_conv_layers
        self.lig_to_rec_conv_layers = lig_to_rec_conv_layers
        self.rec_to_lig_conv_layers = rec_to_lig_conv_layers

        self.center_distance_expansion = GaussianSmearing(0.0, self.center_max_distance, self.distance_embed_dim)
        self.center_edge_embedding = MLP(self.ns, self.ns, self.dropout)

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * self.ns,
            residual=False,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        )

        self.tr_final_layer = MLP(self.ns, 1, self.dropout)
        self.rot_final_layer = MLP(self.ns, 1, self.dropout)

        if not self.no_torsion:
            # torsion angles components
            self.final_edge_embedding = MLP(self.ns, self.ns, self.dropout)
            final_tp_tor = e3nn.tensor_product(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=final_tp_tor,
                out_irreps=f'{self.ns}x0o + {self.ns}x0e',
                n_edge_features=3 * self.ns,
                residual=False,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            )

            self.tor_final_layer = MLP(self.ns, 1, self.dropout, False)

    def __call__(self, inputs, training):

        tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[inputs['complex_t'][noise_type] for noise_type in ['tr', 'rot', 'tor']])

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(inputs)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr, training)
        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(inputs)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr, training)

        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20)[:, None]
        else:
            cross_cutoff = self.cross_max_distance
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(inputs, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr, training)

        for l in range(len(self.lig_conv_layers)):
            lig_edge_attr_ = jnp.concatenate([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], axis=-1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

            # inter graph message passing
            rec_to_lig_edge_attr_ = jnp.concatenate([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], axis=-1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0])
            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = jnp.concatenate([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

                lig_to_rec_edge_attr_ = jnp.concatenate([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], axis=-1)
                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, jnp.flip(cross_edge_index, axis=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0])
            # padding original features
            lig_node_attr = jnp.pad(lig_node_attr, ((0,0),(0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1])))
            # update features with residual updates

            lig_node_attr = lig_node_attr + lig_intra_update.array + lig_inter_update.array

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = jnp.pad(rec_node_attr, ((0,0),(0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1])))
                rec_node_attr = rec_node_attr + rec_intra_update.array + rec_inter_update.array

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(inputs)
        center_edge_attr = self.center_edge_embedding(center_edge_attr, training)
        center_edge_attr = jnp.concatenate([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=inputs.num_graphs).array

        tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
        rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
        inputs.graph_sigma_emb = self.timestep_emb_func(inputs['complex_t']['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = jnp.linalg.norm(tr_pred, axis=1, keepdims=True)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(jnp.concatenate([tr_norm, inputs.graph_sigma_emb], axis=1), training)
        rot_norm = jnp.linalg.norm(rot_pred, axis=1, keepdims=True)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(jnp.concatenate([rot_norm, inputs.graph_sigma_emb], axis=1), training)

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma[:, None]
            rot_pred = rot_pred * so3.score_norm(rot_sigma)[:, None]

        if self.no_torsion or jnp.sum(inputs['ligand'].edge_mask) == 0:
            return tr_pred, rot_pred, jnp.empty((0,))

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_bond_conv_graph(inputs, training)
        tor_bond_vec = inputs['ligand'].pos[tor_bonds[1]] - inputs['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = e3nn.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh.irreps = self.sh_irreps

        tor_edge_sh = e3nn.tensor_product(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]]) #self.sh_irreps, "2e"

        tor_edge_attr = jnp.concatenate([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns], tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh, out_nodes=jnp.sum(inputs['ligand'].edge_mask), reduce='mean')
        tor_pred = self.tor_final_layer(tor_pred.array, training).squeeze(1)

        edge_sigma = tor_sigma[inputs['ligand'].batch][inputs['ligand_lig_bond_ligand'].edge_index[0]][inputs['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * jnp.sqrt(torus.score_norm(edge_sigma))

        return (tr_pred, rot_pred, tor_pred)


    def build_lig_conv_graph(self, inputs):
        inputs['ligand'].node_sigma_emb = self.timestep_emb_func(inputs['ligand'].node_t['tr'])

        # compute edges
        radius_edges = radius_graph(inputs['ligand'].pos, self.lig_max_radius, inputs['ligand'].batch)
        edge_index = jnp.concatenate([inputs['ligand_lig_bond_ligand'].edge_index, radius_edges], axis=1).astype(jnp.int32)
        edge_attr = jnp.concatenate([
            inputs['ligand_lig_bond_ligand'].edge_attr,
            jnp.zeros((radius_edges.shape[-1], self.in_lig_edge_features))
        ], axis=0)

        # compute initial features
        edge_sigma_emb = inputs['ligand'].node_sigma_emb[edge_index[0].astype(jnp.int32)]
        edge_attr = jnp.concatenate([edge_attr, edge_sigma_emb], axis=1)
        node_attr = jnp.concatenate([inputs['ligand'].x, inputs['ligand'].node_sigma_emb], axis=1)

        src, dst = edge_index
        edge_vec = inputs['ligand'].pos[dst.astype(jnp.int32)] - inputs['ligand'].pos[src.astype(jnp.int32)]
        edge_length_emb = self.lig_distance_expansion(jnp.linalg.norm(edge_vec, axis=-1))

        edge_attr = jnp.concatenate([edge_attr, edge_length_emb], axis=1)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_rec_conv_graph(self, inputs):
        inputs['receptor'].node_sigma_emb = self.timestep_emb_func(inputs['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        node_attr = jnp.concatenate([inputs['receptor'].x, inputs['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = inputs['receptor_rec_contact_receptor'].edge_index
        src, dst = edge_index
        edge_vec = inputs['receptor'].pos[dst.astype(jnp.int32)] - inputs['receptor'].pos[src.astype(jnp.int32)]

        edge_length_emb = self.rec_distance_expansion(jnp.linalg.norm(edge_vec, axis=-1))
        edge_sigma_emb = inputs['receptor'].node_sigma_emb[edge_index[0].astype(jnp.int32)]
        edge_attr = jnp.concatenate([edge_sigma_emb, edge_length_emb], axis=1)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh

    def build_cross_conv_graph(self, inputs, cross_distance_cutoff):
        if isinstance(cross_distance_cutoff, jnp.ndarray):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(inputs['receptor'].pos / cross_distance_cutoff[inputs['receptor'].batch],
                                    inputs['ligand'].pos / cross_distance_cutoff[inputs['ligand'].batch], 1,
                                    inputs['receptor'].batch, inputs['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(inputs['receptor'].pos, inputs['ligand'].pos, cross_distance_cutoff,
                                    inputs['receptor'].batch, inputs['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = inputs['receptor'].pos[dst.astype(jnp.int32)] - inputs['ligand'].pos[src.astype(jnp.int32)]

        edge_length_emb = self.cross_distance_expansion(jnp.linalg.norm(edge_vec, axis=-1))

        edge_sigma_emb = inputs['ligand'].node_sigma_emb[src.astype(jnp.int32)]
        edge_attr = jnp.concatenate([edge_sigma_emb, edge_length_emb], axis=1)

        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_center_conv_graph(self, inputs):
        # builds the filter and edges for the convolution generating translational and rotational scores
        edge_index = jnp.concatenate([inputs['ligand'].batch[None, :], jnp.arange(len(inputs['ligand'].batch))[None, :]], axis=0)

        center_pos = jnp.zeros((inputs.num_graphs, 3))
        # center_pos = jnp.add.at(center_pos, inputs['ligand'].batch, inputs['ligand'].pos, inplace=False)
        for i, idx in enumerate(inputs['ligand'].batch):
            center_pos = center_pos.at[idx].add(inputs['ligand'].pos[i])

        center_pos = center_pos / jnp.bincount(inputs['ligand'].batch)[:, None]

        edge_vec = inputs['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(jnp.linalg.norm(edge_vec, axis=-1))
        edge_sigma_emb = inputs['ligand'].node_sigma_emb[edge_index[1].astype(jnp.int32)]
        edge_attr = jnp.concatenate([edge_attr, edge_sigma_emb], axis=1)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, inputs, training):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        bonds = inputs['ligand_lig_bond_ligand'].edge_index[:, inputs['ligand'].edge_mask].astype(jnp.int32)
        bond_pos = (inputs['ligand'].pos[bonds[0]] + inputs['ligand'].pos[bonds[1]]) / 2
        bond_batch = inputs['ligand'].batch[bonds[0]]
        edge_index = radius(inputs['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=inputs['ligand'].batch, batch_y=bond_batch)

        edge_vec = inputs['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(jnp.linalg.norm(edge_vec, axis=-1))

        edge_attr = self.final_edge_embedding(edge_attr, training)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh