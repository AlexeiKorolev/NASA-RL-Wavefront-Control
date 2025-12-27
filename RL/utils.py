import os
import json
import glob
import numpy as np
from environment import CoronagraphEnvironment
import copy
from archs.fc1 import FC1
import torch
import tqdm
from train_job import NORMALIZERS

def load_model(path: str):
    # loading the model in

    metrics_path = path + r'\metrics.json'

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

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
    
    state_obj = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state_obj, dict):
        if 'state_dict' in state_obj:
            state_dict = state_obj['state_dict']
        elif 'model_state_dict' in state_obj:
            state_dict = state_obj['model_state_dict']
        else:
            state_dict = state_obj
    else:
        state_dict = state_obj

    fc1_args = metrics_data['args']
    requested_encoder = fc1_args.get('fc1_encoder_enabled', False)
    state_has_encoder = any(k.startswith('encoder.') for k in state_dict.keys())
    if state_has_encoder != requested_encoder:
        if state_has_encoder:
            print("Checkpoint includes encoder weights; enabling encoder to match saved model.")
        else:
            print("Checkpoint has no encoder weights; disabling encoder to match saved model.")
        encoder_enabled = state_has_encoder
    else:
        encoder_enabled = requested_encoder
    model = FC1(
        final_output_dim=metrics_data['output_dim'], 
        image_input_shape=metrics_data['input_shape'], 
        hidden_layers=fc1_args['fc1_hidden'], 
        activation=fc1_args['fc1_activation'],
        final_activation=fc1_args['fc1_final_activation'],
        dropout=fc1_args.get('fc1_dropout', 0.0),
        encoder_enabled=encoder_enabled,
        filter_sizes=fc1_args.get('fc1_filter_sizes', None),
        filter_channels=fc1_args.get('fc1_filter_channels', None),
        final_embedding_size=fc1_args.get('fc1_final_embedding_size', None),
        final_embedding_channels=fc1_args.get('fc1_final_embedding_channels', None),
        )
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    model.eval()

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
    env_config['num_noise_modes'] = 500
    env = CoronagraphEnvironment(**env_config)

    return model, env, metrics_data


def get_model_history(metrics_data: dict):
    history = metrics_data['history']
    train = history.get('train_loss', [])
    val = history.get('val_loss', [])

    return train, val

def test_noise_levels(model: torch.nn.Module, env: CoronagraphEnvironment, metrics_data: dict, noise_levels: list, noise_mode_amounts: list, delta_t: float = 1e-3, repetitions: int = 100, loops: int = 1):
    print(f"env.num_noise_modes: {env.num_noise_modes}")
    from train_job import NORMALIZERS
    model.eval()
    split_vector = metrics_data['args'].get('split_vector', False)
    print(f"Using split vector: {split_vector}")

    norm_func_name = metrics_data['x_norm']['type']
    if norm_func_name == "log+zscore": norm_func_name = "log"

    x_norm_func = NORMALIZERS[norm_func_name]
    params = metrics_data['x_norm'].copy()
    params.pop('type')

    y_mean = metrics_data['y_norm']['mean']
    y_std = metrics_data['y_norm']['std']

    num_modes = metrics_data['dataset_meta']['environment_config']['num_modes']

    noise_levels = np.atleast_1d(np.asarray(noise_levels, dtype=float))
    noise_mode_amounts = np.atleast_1d(np.asarray(noise_mode_amounts, dtype=int))
    samples = noise_levels.shape[0]
    mode_choices = noise_mode_amounts.shape[0]
    repetitions = int(repetitions)
    loops = int(max(1, loops))
    
    cos_history = np.zeros((samples, mode_choices, repetitions, loops))
    contrast_history = np.zeros((samples, mode_choices, repetitions, loops + 1))
    predictions = np.zeros((samples, mode_choices, repetitions, num_modes))
    truths = np.zeros((samples, mode_choices, repetitions, num_modes))
    
    model_device = next(model.parameters()).device
    
    def prepare_tensor(imgs):
        imgs = np.asarray(imgs, dtype=np.float32)
        imgs = np.expand_dims(imgs, axis=0)
        imgs_norm, _ = x_norm_func(imgs, **params)
        
        tensor = torch.from_numpy(np.transpose(imgs_norm, (0, 2, 3, 1))).float()
        return tensor.to(model_device)
    
    def get_model_output(imgs):
        with torch.no_grad():
            input_tensor = prepare_tensor(imgs)
            y_pred = model(input_tensor)
            y_pred = y_pred.squeeze(0).detach().cpu().numpy()
            if split_vector:
                vec_part = y_pred[:-1]
                work_pred = vec_part / (np.linalg.norm(vec_part) + 1e-12)
            else:
                work_pred = y_pred
        return work_pred * y_std + y_mean
    
    for i, noise in enumerate(tqdm.tqdm(noise_levels, desc="noise sweep")):
        for m_idx, num_noise_modes in enumerate(noise_mode_amounts):
            for j in range(repetitions):
                env.deformable_mirror.flatten()
                env.noise_gen_mirror.flatten()
                env.set_random_noise_dm(noise=noise, num_modes=num_noise_modes)

                initial_dm = np.array(env.deformable_mirror.actuators.copy())
                initial_contrast = env.get_contrast(delta_t=1e15)
                contrast_history[i, m_idx, j, 0] = np.log10(initial_contrast)
                
                for loop_idx in range(loops):
                    original = np.array(env.noise_gen_mirror.actuators.copy())[:num_modes]
                    state_before = np.array(env.deformable_mirror.actuators.copy())
                    imgs = env.generate_diversity_images(delta_t=delta_t, noise_enabled=False)
                    work_pred = get_model_output(imgs)
                    
                    # if split_vector and alpha_lookup is not None:
                    #     alpha_value = np.squeeze(alpha_lookup[i]) if alpha_lookup.ndim > 1 else alpha_lookup[i]
                    #     work_pred = work_pred * alpha_value
                    
                    net_action = state_before - work_pred
                    env.deformable_mirror.flatten()
                    env.deformable_mirror.actuators = net_action

                    cos_num = np.dot(-net_action, original)
                    cos_den = (np.linalg.norm(net_action) * np.linalg.norm(original)) + 1e-40
                    cos_history[i, m_idx, j, loop_idx] = cos_num / cos_den
                    
                    updated_contrast = env.get_contrast(delta_t=1e15)
                    contrast_history[i, m_idx, j, loop_idx + 1] = np.log10(updated_contrast)
                
                truths[i, m_idx, j, :] = initial_dm
                predictions[i, m_idx, j, :] = initial_dm - env.deformable_mirror.actuators
            
    cos_similarities_raw = cos_history[:, :, :, -1]
    cos_similarities = np.mean(cos_similarities_raw, axis=2)
    cos_similarities_std = np.std(cos_similarities_raw, axis=2)
    cos_similarities_25 = np.percentile(cos_similarities_raw, 25, axis=2)
    cos_similarities_75 = np.percentile(cos_similarities_raw, 75, axis=2)
    
    initial_contrasts_raw = contrast_history[:, :, :, 0]
    final_contrasts_raw = contrast_history[:, :, :, -1]
    initial_contrasts = np.mean(initial_contrasts_raw, axis=2)
    initial_contrasts_std = np.std(initial_contrasts_raw, axis=2)
    final_contrasts = np.mean(final_contrasts_raw, axis=2)
    final_contrasts_std = np.std(final_contrasts_raw, axis=2)
    
    evaluation_stats = {
        "noise_levels": noise_levels.tolist(),
        "noise_mode_amounts": noise_mode_amounts.tolist(),
        "loops": loops,
        "repetitions": repetitions,
        "cosine_similarity": {
            "mean": cos_similarities.tolist(),
            "std": cos_similarities_std.tolist(),
            "p25": cos_similarities_25.tolist(),
            "p75": cos_similarities_75.tolist(),
        },
        "initial_contrast_log10": {
            "mean": initial_contrasts.tolist(),
            "std": initial_contrasts_std.tolist()
        },
        "final_contrast_log10": {
            "mean": final_contrasts.tolist(),
            "std": final_contrasts_std.tolist()
        }
    }

    return evaluation_stats, predictions, truths