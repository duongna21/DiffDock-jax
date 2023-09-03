import math
from typing import Optional, Any

import e3nn_jax as e3nn
import jax
import flax.linen as nn
# from torch_cluster import radius, radius_graph
from utils.scatter import scatter, scatter_mean
import numpy as np
import jax.numpy as jnp
from batchnorm_flax import BatchNorm
from f_tp_flax import FullTensorProduct
from fc_tp_flax import FullyConnectedTensorProduct
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims


class GaussianSmearing(nn.Module):
    start: float = 0.0
    stop: float = 5.0
    num_gaussians: int = 50

    def setup(self):
        offset = jnp.linspace(self.start, self.stop, self.num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]) ** 2
        self.offset = offset 

    def __call__(self, dist):
        dist = dist.reshape((-1, 1)) - self.offset.reshape((1, -1))
        return jnp.exp(self.coeff * jnp.power(dist, 2))


class AtomEncoder(nn.Module):
    emb_dim: int
    feature_dims: tuple
    sigma_embed_dim: int
    lm_embedding_type: str = None
    
    def setup(self):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        self.atom_embedding_list = []
        self.num_categorical_features = len(self.feature_dims[0])
        self.num_scalar_features = self.feature_dims[1] + self.sigma_embed_dim
        for _, dim in enumerate(self.feature_dims[0]):
            emb = nn.Embed(num_embeddings=dim, features=self.emb_dim, embedding_init=jax.nn.initializers.glorot_uniform())
            self.atom_embedding_list.append(emb)

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
    n_edge_features: int
    residual: bool = True
    batch_norm: bool = True
    dropout: float = 0.0
    hidden_features: int = None
    
    def setup(self):
        if self.hidden_features is None:
            self.hidden_features = self.n_edge_features

        self.tp = tp = FullyConnectedTensorProduct(self.in_irreps, self.sh_irreps, self.out_irreps)
        self.fc = nn.Sequential([nn.Dense(features=self.hidden_features),
                                nn.relu(),
                                nn.Dropout(rate=self.dropout),
                                nn.Dense(features=tp.weight_numel)])  #TODO int: weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)
        
        self.batch_norm = BatchNorm(self.out_irreps) if self.batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        
        if self.residual:
            # padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            padded = jnp.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]), mode='constant') #TODO check for similarity
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class DiffDock(nn.Module):
    t_to_sigma: Any
    timestep_emb_func: Any
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
    confidence_dropout: float = 0.0
    confidence_no_batchnorm: bool = False 


    def setup(self):
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(lmax=self.sh_lmax)
        self.lig_node_embedding = AtomEncoder(emb_dim=self.ns, feature_dims=lig_feature_dims, sigma_embed_dim=self.sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential([nn.Dense(self.ns),
                                                nn.relu(),
                                                nn.Dropout(self.dropout),
                                                nn.Dense(self.ns)])

        self.rec_node_embedding = AtomEncoder(emb_dim=self.ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=self.sigma_embed_dim, lm_embedding_type=self.lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential([nn.Dense(self.ns),
                                                nn.relu(),
                                                nn.Dropout(self.dropout),
                                                nn.Dense(self.ns)])
        self.cross_edge_embedding = nn.Sequential([nn.Dense(self.ns),
                                                    nn.relu(),
                                                    nn.Dropout(self.dropout),
                                                    nn.Dense(self.ns)])
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
                f'{self.ns}x0e + {self.nv}x1o + {self.nvnv}x1e + {self.ns}x0o'
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
        self.center_edge_embedding = nn.Sequential([nn.Dense(self.ns),
                                                    nn.relu(),
                                                    nn.Dropout(self.dropout),
                                                    nn.Dense(self.ns)])

        self.final_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'2x1o + 2x1e',
            n_edge_features=2 * self.ns,
            residual=False,
            dropout=self.dropout,
            batch_norm=self.batch_norm
        )
        self.tr_final_layer = nn.Sequential([nn.Dense(self.ns),
                                            nn.Dropout(self.dropout), 
                                            nn.relu(), 
                                            nn.Dense(1)])
        self.rot_final_layer = nn.Sequential([nn.Dense(self.ns),
                                             nn.Dropout(self.dropout), 
                                             nn.relu(), 
                                             nn.Dense(1)])

        if not self.no_torsion:
            # torsion angles components
            self.final_edge_embedding = nn.Sequential([nn.Dense(self.ns),
                                                        nn.relu(),
                                                        nn.Dropout(self.dropout),
                                                        nn.Dense(self.ns)])
            self.final_tp_tor = FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f'{self.ns}x0o + {self.ns}x0e',
                n_edge_features=3 * self.ns,
                residual=False,
                dropout=self.dropout,
                batch_norm=self.batch_norm
            )
            self.tor_final_layer = nn.Sequential([nn.Dense(self.ns, use_bias=False),
                                                    jnp.tanh(),
                                                    nn.Dropout(self.dropout),
                                                    nn.Dense(1, use_bias=False)])
            
    def __call__(self, input):
        pass

    def build_lig_conv_graph(self, input):
        pass

    def build_rec_conv_graph(self, input):
        pass
    
    def build_cross_conv_graph(self, input):
        pass
    
    def build_center_conv_graph(self, input):
        pass
    
    def build_bond_conv_graph(self, input):
        pass