import jax.numpy as jnp
from jax import jit, vmap
from flax import linen as nn
from utils import so3, torus
import jax
from models.score_model import DiffDock
from utils.diffusion_utils import get_timestep_embedding
import optax
from typing import Optional, Any, Callable
from tqdm import tqdm

def loss_function(tr_pred, rot_pred, tor_pred, data, t_to_sigma, tr_weight=1, rot_weight=1, tor_weight=1, apply_mean=True, no_torsion=False):
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(*[jnp.concatenate([d.complex_t[noise_type] for d in data])
                                                for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component
    tr_score = jnp.concatenate([d.tr_score for d in data], axis=0)
    tr_sigma = tr_sigma[:, None]
    tr_loss = jnp.mean(((tr_pred - tr_score) ** 2 * tr_sigma ** 2), axis=mean_dims)
    tr_base_loss = jnp.mean((tr_score ** 2 * tr_sigma ** 2),axis=mean_dims)

    # rotation component
    rot_score = jnp.concatenate([d.rot_score for d in data], axis=0)
    rot_score_norm = so3.score_norm(rot_sigma)[:, None]
    rot_loss = jnp.mean((((rot_pred - rot_score) / rot_score_norm) ** 2),axis=mean_dims)
    rot_base_loss = jnp.mean(((rot_score / rot_score_norm) ** 2),axis=mean_dims)

    # torsion component
    if not no_torsion:
        edge_tor_sigma = jnp.concatenate([d.tor_sigma_edge for d in data])
        tor_score = jnp.concatenate([d.tor_score for d in data], axis=0)
        tor_score_norm2 = torus.score_norm(edge_tor_sigma)
        tor_loss = ((tor_pred - tor_score) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score ** 2 / tor_score_norm2))

        if apply_mean:
            tor_loss = jnp.mean(tor_loss)
            tor_base_loss = jnp.mean(tor_base_loss)
        else:
            index = jnp.concatenate([jnp.ones(d['ligand'].edge_mask.sum()) * i for i, d in enumerate(data)])
            num_graphs = len(data)
            t_l, t_b_l, c = jnp.zeros(num_graphs), jnp.zeros(num_graphs), jnp.zeros(num_graphs)
            c = jnp.add.at(c, index, jnp.ones(tor_loss.shape), inplace=False)
            c += 0.0001
            t_l = jnp.add.at(t_l, index, tor_loss, inplace=False)
            t_b_l = jnp.add.at(t_b_l, index, tor_base_loss, inplace=False)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = jnp.zeros(1), jnp.zeros(1)
        else:
            tor_loss, tor_base_loss = jnp.zeros(len(rot_loss)), jnp.zeros(len(rot_loss))

    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight
    return loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss


def create_learning_rate_fn(config, num_train_steps) -> Callable[[int], jnp.array]:
    """Create the learning rate function."""
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps + 1,  # ensure not 0
    )
    last_boundary = config.warmup_steps
    # offset step when resuming
    if config.lr_offset:
        warmup_fn = optax.join_schedules(
            schedules=[optax.constant_schedule(0.0), warmup_fn],
            boundaries=[config.lr_offset],
        )
        last_boundary += config.lr_offset
    if config.lr_decay is None:
        return warmup_fn
    elif config.lr_decay == "linear":
        assert (num_train_steps is not None), "linear decay requires knowing the dataset length"
        decay_fn = optax.linear_schedule(
            init_value=config.learning_rate,
            end_value=0,
            transition_steps=num_train_steps - config.warmup_steps,
        )
    elif config.lr_decay == "exponential":
        decay_fn = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.lr_transition_steps,
            decay_rate=config.lr_decay_rate,
            staircase=config.lr_staircase,
        )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[last_boundary],
    )
    return schedule_fn


def get_model(config, t_to_sigma):
    timestep_emb_func = get_timestep_embedding(
        embedding_type=config.embedding_type,
        embedding_dim=config.sigma_embed_dim,
        embedding_scale=config.embedding_scale)

    lm_embedding_type = None
    if config.esm_embeddings_path is not None: lm_embedding_type = 'esm'

    model = DiffDock(t_to_sigma=t_to_sigma,
                        no_torsion=config.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=config.num_conv_layers,
                        lig_max_radius=config.max_radius,
                        scale_by_sigma=config.scale_by_sigma,
                        sigma_embed_dim=config.sigma_embed_dim,
                        ns=config.ns, nv=config.nv,
                        distance_embed_dim=config.distance_embed_dim,
                        cross_distance_embed_dim=config.cross_distance_embed_dim,
                        batch_norm=not config.no_batch_norm,
                        dropout=config.dropout,
                        use_second_order_repr=config.use_second_order_repr,
                        cross_max_distance=config.cross_max_distance,
                        dynamic_max_cross=config.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type,
                        )

    return model


class Metrics():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else jnp.zeros((len(types), intervals))
        self.acc = {t: jnp.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].ndim == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum() if self.unpooled_metrics else v
        else:
            for type_idx, v in enumerate(vals):
                #self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                temp = self.count[type_idx]
                self.count = self.count.at[type_idx].set(jnp.add.at(temp, interval_idx[type_idx], jnp.ones(len(v))), inplace=False)   #TODO
                if not jnp.allclose(v, jnp.array(0.0)): 
                    #self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)
                    temp = self.acc[self.types[type_idx]]
                    self.acc[self.types[type_idx]] = jnp.add.at(temp, interval_idx[type_idx], v, inplace=False)
        return self

    def summary(self):
        if self.intervals == 1:
            out = {k: v / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = list(self.acc.values())[type_idx][i] / self.count[type_idx][i]
            return out



def test_epoch(model, loader, device, t_to_sigma, loss_fn, state, params, test_sigma_intervals=False):
    meter = Metrics(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
                         unpooled_metrics=True)

    if test_sigma_intervals:
        meter_all = Metrics(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'],
            unpooled_metrics=True, intervals=10)

    for data in tqdm(loader, total=len(loader)):
        (tr_pred, rot_pred, tor_pred), _ = model.apply({'params': params, **state}, data, mutable=['batch_stats'], training=False)

        loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = \
            loss_fn(tr_pred, rot_pred, tor_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
        meter.add([loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])

        if test_sigma_intervals > 0:
            complex_t_tr, complex_t_rot, complex_t_tor = [jnp.concatenate([d.complex_t[noise_type] for d in data]) for
                                                            noise_type in ['tr', 'rot', 'tor']]
            sigma_index_tr = jnp.round(complex_t_tr * (10 - 1)).astype(jnp.int32)
            sigma_index_rot = jnp.round(complex_t_rot * (10 - 1)).astype(jnp.int32)
            sigma_index_tor = jnp.round(complex_t_tor * (10 - 1)).astype(jnp.int32)
            meter_all.add(
                [loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss],
                [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_rot,
                    sigma_index_tor, sigma_index_tr])

    out = meter.summary()
    if test_sigma_intervals > 0: 
        out.update(meter_all.summary())
        
    return out
