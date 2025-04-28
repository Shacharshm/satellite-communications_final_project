import gzip
import pickle
import pprint
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.models import load_model
from matplotlib.pyplot import show as plt_show

from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.config.config import Config
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.models.helpers.learned_precoder import get_learned_precoder_normalized
from src.data.calc_sum_rate import calc_sum_rate
from src.utils.plot_beampattern import plot_beampattern
from src.utils.update_sim import update_sim


plot = [
    'mmse',
    'slnr',
    'learned',
    # 'ones',
]

# angle_sweep_range = np.arange((90 - 30) * np.pi / 180, (90 + 30) * np.pi / 180, 0.1 * np.pi / 180)  # arange or None
angle_sweep_range = np.arange(1.2, 1.9, 0.1 * np.pi / 180)  # arange or None


config = Config()
# config.user_dist_bound = 0  # disable user wiggle
# config.user_dist_bound = 50_000

model_path = Path(  # SAC only
    config.trained_models_path,
    '1_sat_16_ant_3_usr_10000_dist_0.0_error_on_cos_0.1_fading',
    'single_error',
    'userwiggle_5000_snap_3.926',
    'model',
)

if 'learned' in plot:
    from src.utils.compare_configs import compare_configs
    compare_configs(config, Path(model_path, '..', 'config'))

    with tf.device('CPU:0'):

        with gzip.open(Path(model_path, '..', 'config', 'norm_dict.gzip')) as file:
            norm_dict = pickle.load(file)
        norm_factors = norm_dict['norm_factors']
        if norm_factors != {}:
            config.config_learner.get_state_args['norm_state'] = True
        else:
            config.config_learner.get_state_args['norm_state'] = False

        try:
            precoder_network = tf.saved_model.load(str(model_path))
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying alternative loading method...")
            try:
                precoder_network = load_model(model_path)
            except Exception as e:
                print(f"Failed to load model with both methods: {str(e)}")
                raise

satellite_manager = SatelliteManager(config)
user_manager = UserManager(config)

def save_results(results, config, model_path):
    """Save the beam pattern results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.output_metrics_path, 'beam_pattern_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results
    results_file = results_dir / f'beam_pattern_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Save plots
    plots_dir = results_dir / f'plots_{timestamp}'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    return results_file, plots_dir

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

results = {
    'config': {
        'angle_sweep_range': angle_sweep_range.tolist(),
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat()
    },
    'iterations': []
}

# Create results directory first
results_file, plots_dir = save_results(results, config, model_path)

for iter_id in range(2):
    iter_data = {
        'iteration': iter_id,
        'estimation_errors': {},
        'sum_rates': {},
        'user_positions': []
    }

    update_sim(config=config, satellite_manager=satellite_manager, user_manager=user_manager)
    for satellite in satellite_manager.satellites:
        # Convert estimation_errors dictionary to a serializable format
        iter_data['estimation_errors'][f'satellite_{satellite.idx}'] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in satellite.estimation_errors.items()
        }
        iter_data['user_positions'] = [float(pos) for pos in satellite.aods_to_users]

    # MMSE
    if 'mmse' in plot:
        w_mmse = mmse_precoder_normalized(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            **config.mmse_args,
        )

        sum_rate_mmse = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_mmse,
            noise_power_watt=config.noise_power_watt,
        )

        iter_data['sum_rates']['mmse'] = float(sum_rate_mmse)
        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_mmse,
            plot_title='mmse',
            angle_sweep_range=angle_sweep_range,
            save_path=str(plots_dir / f'mmse_iter_{iter_id}.png')
        )

    # SLNR
    if 'slnr' in plot:
        autocorrelation = calc_autocorrelation(
            satellite=satellite_manager.satellites[0],
            error_model_config=config.config_error_model,
            error_distribution='uniform',
        )

        w_slnr = robust_SLNR_precoder_no_norm(
            channel_matrix=satellite_manager.erroneous_channel_state_information,
            autocorrelation_matrix=autocorrelation,
            noise_power_watt=config.noise_power_watt,
            power_constraint_watt=config.power_constraint_watt,
        )

        sum_rate_slnr = calc_sum_rate(
            channel_state=satellite_manager.channel_state_information,
            w_precoder=w_slnr,
            noise_power_watt=config.noise_power_watt,
        )

        iter_data['sum_rates']['slnr'] = float(sum_rate_slnr)
        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_slnr,
            plot_title='slnr',
            angle_sweep_range=angle_sweep_range,
            save_path=str(plots_dir / f'slnr_iter_{iter_id}.png')
        )

    # Learned
    if 'learned' in plot:
        with tf.device('CPU:0'):
            state = config.config_learner.get_state(
                satellite_manager=satellite_manager,
                norm_factors=norm_factors,
                **config.config_learner.get_state_args
            )

            w_learned = get_learned_precoder_normalized(
                state=state,
                precoder_network=precoder_network,
                **config.learned_precoder_args,
            )

            sum_rate_learned = calc_sum_rate(
                channel_state=satellite_manager.channel_state_information,
                w_precoder=w_learned,
                noise_power_watt=config.noise_power_watt,
            )

        iter_data['sum_rates']['learned'] = float(sum_rate_learned)
        plot_beampattern(
            satellite=satellite_manager.satellites[0],
            users=user_manager.users,
            w_precoder=w_learned,
            plot_title='learned',
            angle_sweep_range=angle_sweep_range,
            save_path=str(plots_dir / f'learned_iter_{iter_id}.png')
        )

    results['iterations'].append(iter_data)

# Save final results
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4, cls=NumpyEncoder)
print(f"Results saved to: {results_file}")
print(f"Plots saved to: {plots_dir}")

plt_show()
