from environment import CoronagraphEnvironment
import hcipy
from hcipy import *
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
import pickle
import json
from pprint import pprint
import argparse as ap
from dataclasses import dataclass, asdict
import tqdm
from data_generator import save_dataset
from train_job import NORMALIZERS


if __name__ == "__main__":
    ap = ap.ArgumentParser()
    ap.add_argument('--model_name', type=str, default="linear model", help='Name of the model to evaluate.')
    ap.add_argument('--repetitions', type=int, default=10, help='Number of repetitions per noise level.')
    ap.add_argument('--noise_level_start', type=float, default=1e-8, help='Starting noise level.')
    ap.add_argument('--noise_level_end', type=float, default=1e-6, help='Ending noise level.')
    ap.add_argument('--num_noise_levels', type=int, default=500, help='Number of noise levels to evaluate.')
    args = ap.parse_args()


    paths_path = r"info\paths.json"
    with open(paths_path, 'r') as f:
        paths = json.load(f)

    model_name = args.model_name

    path = paths[model_name]

    metrics_path = path + r'\metrics.json'

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

    # loading the model in

    import glob
    import numpy as np

    # search for .pth/.pt files in the experiment directory (uses existing `path` variable)
    candidates = glob.glob(os.path.join(path, '*.pth')) + glob.glob(os.path.join(path, '*.pt'))
    if not candidates:
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(('.pth', '.pt')):
                    candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError(f'No .pth/.pt files found under {path!r}')

    checkpoint_path = max(candidates, key=os.path.getmtime)

    if metrics_data['args']['fc1_final_activation'] != 'none':
        raise DeprecationWarning("This model has a poor final activation and is deprecated for evaluation.")
    
    from archs.fc1 import FC1
    import torch

    model = FC1(
        final_output_dim=metrics_data['output_dim'], 
        image_input_shape=metrics_data['input_shape'], 
        hidden_layers=metrics_data['args']['fc1_hidden'], 
        activation=metrics_data['args']['fc1_activation'],
        final_activation=metrics_data['args']['fc1_final_activation'],
        dropout=metrics_data['args'].get('fc1_dropout', 0.0),
        encoder_enabled=metrics_data['args'].get('fc1_encoder_enabled', False),
        filter_sizes=metrics_data['args'].get('fc1_filter_sizes', None),
        filter_channels=metrics_data['args'].get('fc1_filter_channels', None),
        final_embedding_size=metrics_data['args'].get('fc1_final_embedding_size', None),
        final_embedding_channels=metrics_data['args'].get('fc1_final_embedding_channels', None),
        )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    from environment import CoronagraphEnvironment
    import copy


    def revive_jsonable(obj):
        if isinstance(obj, dict):
            if obj.get("__type") == "slice":
                return slice(obj.get("start"), obj.get("stop"), obj.get("step"))
            return {k: revive_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [revive_jsonable(v) for v in obj]
        return obj


    raw_env_config = metrics_data['dataset_meta']['environment_config']
    env_config = revive_jsonable(copy.deepcopy(raw_env_config))
    if 'pixel_resolution' in env_config:
        env_config['pixels'] = env_config.pop('pixel_resolution')
    if 'ppsr' in env_config:
        env_config['pixels_per_spacial_res'] = env_config.pop('ppsr')
    if 'env_delta_t' in env_config:
        env_config['delta_t'] = env_config.pop('env_delta_t')

    env_config['obs_noise_level'] = 0.0  # disable observation noise for evaluation

    env = CoronagraphEnvironment(**env_config)

    delta_t = 1e-3 # metrics_data['dataset_meta']['dataset_config']['delta_t']
    delta_t *= 100 if model_name != "linear model" else 1 # due to a data scaling error, need this to compensate.
    # dm_random_noise = metrics_data['dataset_meta']['dataset_config']['dm_random_noise']
    train_type = metrics_data['args'].get('train_type', 'images')
    slopes_train = False # (train_type == "slopes")
    split_vector = metrics_data['args'].get('split_vector', False)
    print(f"Using split vector: {split_vector}")
    print(f"Using slopes: {slopes_train}")

    norm_func_name = metrics_data['x_norm']['type']
    if norm_func_name == "log+zscore": norm_func_name = "log"

    x_norm_func = NORMALIZERS[norm_func_name]
    params = metrics_data['x_norm'].copy()
    params.pop('type')

    y_mean = metrics_data['y_norm']['mean']
    y_std = metrics_data['y_norm']['std']

    def get_model_output(model, imgs):
        with torch.no_grad():
            imgs = torch.asarray(imgs).unsqueeze(0)
            imgs_norm, _ = x_norm_func(imgs, **params)
            imgs_norm = np.transpose(imgs_norm, (0, 2, 3, 1))

            y_pred = model(imgs_norm)
            y_pred = y_pred.squeeze(0).numpy()

            if split_vector:
                vec_part = y_pred[:-1]
                magnitude_part = y_pred[-1]
                work_pred = vec_part
                work_pred /= np.linalg.norm(vec_part)
            else:
                work_pred = y_pred
        
        work_pred = work_pred * y_std + y_mean

        return work_pred

    noise_levels = np.linspace(args.noise_level_start, args.noise_level_end, args.num_noise_levels)
    repetitions = args.repetitions

    truths = []
    predictions = []
    images = []
    residuals = []


    for i, noise in enumerate(tqdm.tqdm(noise_levels)):
        initial_contrasts_raw = []
        final_contrasts_raw = []
        cos_similarities_raw = []

        for j in range(repetitions):
            env.deformable_mirror.flatten()
            env.set_random_dm(noise=noise)
            original = np.array(env.deformable_mirror.actuators.copy())
            initial_contrast = env.get_contrast(delta_t=1e15)
            initial_contrasts_raw.append(initial_contrast)

            imgs = env.generate_diversity_images(delta_t=delta_t, noise_enabled=False)

            work_pred = get_model_output(model, imgs)
                
            truths.append(original)
            predictions.append(work_pred)
            images.append(imgs)

            net_action = original - work_pred
            residuals.append(net_action)
    


    meta = {
        "environment_config": raw_env_config,
        "dataset_config": None,
        "num_samples": len(truths),
        "image_triplet_description": "(baseline, +nudge, -nudge)"
    }

    data =  {
        "dm_settings": residuals,
        "images": images,
        "slopes": None,
        "meta": meta
    }



    save_path = os.path.join(path, 'residual_data.pkl')
    save_dataset(data, save_path)
    print(f"Saved residual dataset to {save_path}")


