import math
from typing import Optional

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
# from utils import so3, torus
from datasets.process_mols import lig_feature_dims


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
        self.fc = nn.Sequential(
            nn.Dense(features=self.hidden_features),
            nn.relu(),
            nn.Dropout(rate=self.dropout),
            nn.Dense(features=tp.weight_numel)  #TODO int: weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)
        )
        
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

