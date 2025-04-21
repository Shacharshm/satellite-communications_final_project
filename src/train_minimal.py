from pathlib import Path
import tensorflow as tf
import numpy as np
from datetime import datetime

from src.config.minimal_config import MinimalConfig
from src.models.train_sac import train_sac_single_error
from src.analysis.helpers.test_precoder_error_sweep import test_precoder_error_sweep
from src.analysis.generate_beampatterns import generate_beampatterns
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager

def main():
    # Set up configuration
    config = MinimalConfig()
    
    # Configure training name based on parameters
    config.config_learner.training_name = (
        f'sat_{config.sat_nr}'
        f'_ant_{config.sat_tot_ant_nr}'
        f'_usr_{config.user_nr}'
        f'_dist_{config.user_dist_average}'
    )
    
    # Log start of training
    config.logger.info(f"Starting training with configuration: {config.config_learner.training_name}")
    config.logger.info(f"Training started at: {datetime.now()}")
    
    # Train the model
    try:
        best_model_path = train_sac_single_error(config=config)
        config.logger.info(f"Training completed. Best model saved at: {best_model_path}")
        
        # Test the trained model
        error_sweep_range = np.arange(0.0, 0.6, 0.1)
        test_precoder_error_sweep(
            config=config,
            error_sweep_parameter='additive_error_on_cosine_of_aod',
            error_sweep_range=error_sweep_range,
            precoder_name='sac',
            monte_carlo_iterations=100,
            get_precoder_func=None,  # Will be set up in the function
            calc_sum_rate_func=None  # Will be set up in the function
        )
        
        # Generate beampatterns
        satellite_manager = SatelliteManager(config=config)
        user_manager = UserManager(config=config)
        
        angle_sweep_range = np.arange(1.2, 1.9, 0.1 * np.pi / 180)
        generate_beampatterns(
            angle_sweep_range=angle_sweep_range,
            num_patterns=1000,
            config=config,
            satellite_manager=satellite_manager,
            user_manager=user_manager,
            learned_model_paths={
                'sac_model': best_model_path
            },
            generate_mmse=True,
            generate_slnr=True,
            generate_ones=False
        )
        
        config.logger.info("All analysis completed successfully")
        
    except Exception as e:
        config.logger.error(f"Error during training/analysis: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    # Force CPU usage to ensure reproducibility
    with tf.device('CPU:0'):
        main() 