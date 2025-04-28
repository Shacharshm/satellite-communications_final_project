import gzip
import pickle
import time
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.models import load_model

import src
from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.data.precoder.mmse_precoder import mmse_precoder_normalized
from src.data.precoder.calc_autocorrelation import calc_autocorrelation
from src.data.precoder.robust_SLNR_precoder import robust_SLNR_precoder_no_norm
from src.models.helpers.learned_precoder import get_learned_precoder_normalized
from src.utils.update_sim import update_sim
from src.utils.profiling import start_profiling, end_profiling


def save_runtime_results(results, config, model_path):
    """Save the runtime analysis results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config.output_metrics_path, 'runtime_analysis_results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f'runtime_analysis_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    return results_file

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

def main():

    config = Config()

    num_monte_carlo = 300
    num_precodings_per = 300

    which_precoder = 'slnr'  # mmse, slnr, sac

    model_path = Path(  # sac only
        config.trained_models_path,
        '1_sat_16_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading',
        'single_error',
        'userwiggle_50000_snap_4.565',
        'model',
    )

    results = {
        'config': {
            'num_monte_carlo': num_monte_carlo,
            'num_precodings_per': num_precodings_per,
            'which_precoder': which_precoder,
            'model_path': str(model_path),
            'timestamp': datetime.now().isoformat()
        },
        'execution_times': []
    }

    if which_precoder == 'mmse':
        def precoder(sat_man):
            w_mmse = mmse_precoder_normalized(channel_matrix=sat_man.erroneous_channel_state_information, **config.mmse_args)
            return w_mmse

    elif which_precoder == 'slnr':
        def precoder(sat_man):
            autocorrelation = calc_autocorrelation(satellite=sat_man.satellites[0], error_model_config=config.config_error_model, error_distribution='uniform')
            w_robust_slnr = robust_SLNR_precoder_no_norm(
                channel_matrix=satellite_manager.erroneous_channel_state_information,
                autocorrelation_matrix=autocorrelation,
                noise_power_watt=config.noise_power_watt,
                power_constraint_watt=config.power_constraint_watt,
            )
            return w_robust_slnr

    elif which_precoder == 'sac':

        try:
            model = tf.saved_model.load(str(model_path))
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying alternative loading method...")
            try:
                model = load_model(model_path)
            except Exception as e:
                print(f"Failed to load model with both methods: {str(e)}")
                raise

        with gzip.open(Path(model_path, '..', 'config', 'norm_dict.gzip')) as file:
            norm_dict = pickle.load(file)
        norm_factors = norm_dict['norm_factors']

        def precoder(sat_man):
            state = config.config_learner.get_state(satellite_manager=sat_man, norm_factors=norm_factors, **config.config_learner.get_state_args)
            w_learned = get_learned_precoder_normalized(state=state, precoder_network=model, **config.learned_precoder_args)
            return w_learned

    satellite_manager = SatelliteManager(config=config)
    user_manager = UserManager(config=config)

    update_sim(config, satellite_manager, user_manager)

    execution_times = np.empty(num_monte_carlo)
    profiler = start_profiling()
    now = time.process_time()
    for monte_carlo_id in range(num_monte_carlo):
        iter_times = []

        for iter_id in range(num_precodings_per):
            update_sim(config, satellite_manager, user_manager)
            iter_start = time.process_time()
            w_precoder = precoder(satellite_manager)
            iter_end = time.process_time()
            iter_times.append(iter_end - iter_start)

        now_2 = time.process_time()
        execution_times[monte_carlo_id] = now_2 - now
        now = now_2

        results['execution_times'].append({
            'monte_carlo_id': monte_carlo_id,
            'total_time': float(execution_times[monte_carlo_id]),
            'average_time_per_iteration': float(np.mean(iter_times)),
            'std_time_per_iteration': float(np.std(iter_times)),
            'min_time_per_iteration': float(np.min(iter_times)),
            'max_time_per_iteration': float(np.max(iter_times))
        })

    execution_times /= num_precodings_per

    end_profiling(profiler)
    
    # Add summary statistics
    results['summary'] = {
        'mean_execution_time': float(np.mean(execution_times)),
        'std_execution_time': float(np.std(execution_times)),
        'min_execution_time': float(np.min(execution_times)),
        'max_execution_time': float(np.max(execution_times))
    }

    # Save results
    results_file = save_runtime_results(results, config, model_path)
    print(f"Results saved to: {results_file}")
    print(f'{which_precoder} mean run time: {np.mean(execution_times):.5f} +- {np.std(execution_times):.5f}')


if __name__ == '__main__':
    with tf.device('CPU:0'):
        main()
