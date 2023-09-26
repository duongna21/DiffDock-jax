import jax.numpy as jnp
import numpy as np
from utils.torsion import modify_conformer_torsion_angles
import jax
from jax.scipy.spatial.transform import Rotation as R
from utils.diffusion_utils import modify_conformer, set_time
from jax import random

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max, seed):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = jax.random.uniform(random.PRNGKey(seed), minval=-jnp.pi, maxval=jnp.pi, 
                                                 shape=jnp.sum(complex_graph['ligand'].edge_mask))
            complex_graph['ligand'].pos = modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                            complex_graph['ligand', 'ligand'].edge_index.T[complex_graph['ligand'].edge_mask],
                                            complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = jnp.mean(complex_graph['ligand'].pos, axis=0, keepdims=True)
        
        # Assuming R.random().as_matrix() returns a numpy array
        random_rotation = jnp.array(R.random(random.PRNGKey(seed)).as_matrix())
        
        complex_graph['ligand'].pos = jnp.dot(complex_graph['ligand'].pos - molecule_center, random_rotation.T)

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = tr_sigma_max * random.normal(random.PRNGKey(seed), shape=(1, 3)) #mean + std * N(0,1)
            complex_graph['ligand'].pos += tr_update



def sampling(data_list, model, state, params, seed, inference_steps, tr_schedule, rot_schedule, tor_schedule, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]
        
        dataloader = [data_list[i:i+batch_size] for i in range(0, N, batch_size)] #TODO
        new_data_list = []

        for complex_graph_batch in dataloader:
            b = len(complex_graph_batch)

            tr_sigma, rot_sigma, tor_sigma = t_to_sigma(t_tr, t_rot, t_tor)
            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms)

            # tr_score, rot_score, tor_score = model(complex_graph_batch)
            (tr_score, rot_score, tor_score), state = model.apply({'params': params, **state}, complex_graph_batch, mutable=['batch_stats'], training=False)

            tr_g = tr_sigma * jnp.sqrt(2 * jnp.log(model_args.tr_sigma_max / model_args.tr_sigma_min))
            rot_g = 2 * rot_sigma * jnp.sqrt(jnp.log(model_args.rot_sigma_max / model_args.rot_sigma_min))
            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score)
                rot_perturb = (0.5 * rot_score * dt_rot * rot_g ** 2)
            else:
                tr_z = jnp.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) else random.normal(random.PRNGKey(seed), (b, 3))
                tr_perturb = tr_g ** 2 * dt_tr * tr_score + tr_g * jnp.sqrt(dt_tr) * tr_z

                rot_z = jnp.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) else random.normal(random.PRNGKey(seed), (b, 3))
                rot_perturb = rot_score * dt_rot * rot_g ** 2 + rot_g * jnp.sqrt(dt_rot) * rot_z

            if not model_args.no_torsion:
                tor_g = tor_sigma * jnp.sqrt(2 * jnp.log(model_args.tor_sigma_max / model_args.tor_sigma_min))
                if ode:
                    tor_perturb = 0.5 * tor_g ** 2 * dt_tor * tor_score
                else:
                    tor_z = jnp.zeros(tor_score.shape) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) else random.normal(random.PRNGKey(seed), tor_score.shape)
                    tor_perturb = tor_g ** 2 * dt_tor * tor_score + tor_g * jnp.sqrt(dt_tor) * tor_z
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None)
                         for i, complex_graph in enumerate(complex_graph_batch)])
        
        data_list = new_data_list

        # if visualization_list is not None:
        #     for idx, visualization in enumerate(visualization_list):
        #         visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
        #                           part=1, order=t_idx + 2)

    return data_list
