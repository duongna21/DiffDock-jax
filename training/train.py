import jax
import jax.numpy as jnp
import optax
import flax
from models.score_model import DiffDock
from training.train_utils import get_model, loss_function, create_learning_rate_fn, Metrics
from utils.diffusion_utils import t_to_sigma
from functools import partial
from typing import Optional, Any, Callable
from tqdm import tqdm
from flax.training import train_state, checkpoints as flax_checkpoints
from flax.metrics import tensorboard
from absl import logging
from utils.parsing import parse_train_args


def train():
    config = parse_train_args()
    
    #TODO data
    ds_train = None
    batch_size_per_node = None
    
    #define dataset here
    steps_per_epoch = (
        len(ds_train) // batch_size_per_node
        if len(ds_train) is not None
        else None
    )
    num_train_steps = (
        steps_per_epoch * config.num_epochs if steps_per_epoch is not None else None
    )
    
    batch = None
    
    "__________________________________________________________"
    
    model = get_model(config, t_to_sigma)
    
    root_key = jax.random.PRNGKey(seed=0)
    _, params_key, dropout_key = jax.random.split(key=root_key, num=3)
    
    def init_model():
      return model.init(params_key, batch, training=False)
    
    # jax.jit(init_model, backend='cpu')()
    variables = init_model()
    state, params = flax.core.pop(variables, 'params')
    del variables
    
    apply_fn = model.apply
    loss_fn = partial(loss_function, tr_weight=config.tr_weight, rot_weight=config.rot_weight,
                      tor_weight=config.tor_weight, no_torsion=config.no_torsion)

    lr_fn = create_learning_rate_fn(config, num_train_steps)
    
    tx = optax.chain(
      optax.clip_by_global_norm(config.grad_norm_clip),
      optax.adam(
          learning_rate=lr_fn,
          accumulator_dtype='bfloat16',
      ),
    )
    
    initial_step = 1
    opt_state = tx.init(params)
    log_folder = workdir = config.log_folder
    summary_writer = tensorboard.SummaryWriter(log_folder)
    
    # @jax.jit
    def make_update_fn(apply_fn, loss_fn, state, tx):
      def train_step(x, opt_state, params, metric, dropout_key):
        def loss(params, inputs):
          _, new_dropout_key = jax.random.split(dropout_key)
          (tr_pred, rot_pred, tor_pred), state  = apply_fn({'params': params, **state}, inputs, mutable=list(state.keys()), 
                                                  training=True, rngs={'dropout': new_dropout_key})
          
          loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss = loss_fn(tr_pred, rot_pred, tor_pred) 
          metric = metric.add([loss, tr_loss, rot_loss, tor_loss, tr_base_loss, rot_base_loss, tor_base_loss])
          return loss, metric, new_dropout_key

        (l, metric, new_dropout_key), g = jax.value_and_grad(loss, has_aux=True)(params, x)
        updates, opt_state = tx.update(g, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, metric, state, new_dropout_key
      return train_step
    
    update_fn = make_update_fn(apply_fn=model.apply, loss_fn=loss_fn, state=state, tx=tx)
    
    for epoch in tqdm(range(config.num_epochs)):
      metrics = Metrics(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'tr_base_loss', 'rot_base_loss', 'tor_base_loss'])
      for step, batch in zip(tqdm(range(steps_per_epoch)), ds_train.as_numpy_iterator()):
        opt_state, params, metric, state, dropout_key = update_fn(apply_fn, loss_fn, batch, opt_state, params, state, metric, dropout_key)
        summary_writer.scalar('lr', step + epoch*steps_per_epoch, epoch)
        if ((config.checkpoint_every and step % config.eval_every == 0) or step == num_train_steps):
          checkpoint_path = flax_checkpoints.save_checkpoint(
              workdir, (params,opt_state, step), step)
          logging.info('Stored checkpoint at step %d to "%s"', step, checkpoint_path)
          
      loss_dict = metrics.summary()
      summary_writer.scalar('loss', loss_dict['loss'], epoch)
      summary_writer.scalar('tr_loss', loss_dict['tr_loss'], epoch)
      summary_writer.scalar('rot_loss', loss_dict['rot_loss'], epoch)
      summary_writer.scalar('tor_loss', loss_dict['tor_loss'], epoch)
      
      
if __name__ == '__main__':
    train()