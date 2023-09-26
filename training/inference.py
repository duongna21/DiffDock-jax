import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tqdm import tqdm
import copy
from utils.diffusion_utils import get_t_schedule
from utils.sampling import randomize_position, sampling


def inference_epoch(model, complex_graphs, device, t_to_sigma, args):
    t_schedule = get_t_schedule(inference_steps=args.inference_steps)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    # Assuming you have a modified ListDataset and DataLoader for JAX/Flax
    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    rmsds = []

    for orig_complex_graph in tqdm(loader):
        data_list = [copy.deepcopy(orig_complex_graph)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        predictions_list = None
        while predictions_list is None:
            predictions_list = sampling(data_list=data_list, model=model,
                                        inference_steps=args.inference_steps,
                                        tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                        tor_schedule=tor_schedule,
                                        t_to_sigma=t_to_sigma, model_args=args)
        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos + orig_complex_graph.original_center)

        filterHs = jnp.not_equal(predictions_list[0]['ligand'].x[:, 0], 0)

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        ligand_pos = jnp.asarray(
            [complex_graph['ligand'].pos[filterHs] for complex_graph in predictions_list])
        orig_ligand_pos = jnp.expand_dims(
            orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center, axis=0)
        rmsd = jnp.sqrt(jnp.mean(jnp.sum((ligand_pos - orig_ligand_pos) ** 2, axis=2),axis=1))
        rmsds.append(rmsd)

    losses = {'rmsds_lt2': jnp.sum(100 * (rmsds < 2) / len(rmsds)),
              'rmsds_lt5': jnp.sum(100 * (rmsds < 5) / len(rmsds))}
    return losses


