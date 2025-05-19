from pathlib import Path
import numpy as np
from src.analysis.helpers.test_robust_slnr_precoder_user_distance_sweep import test_robust_slnr_precoder_distance_sweep
from src.analysis.helpers.test_mmse_precoder_error_sweep import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_mmse_precoder_user_distance_sweep import test_mmse_precoder_user_distance_sweep
from src.analysis.helpers.test_mrc_precoder_error_sweep import test_mrc_precoder_error_sweep
from src.analysis.helpers.test_mrc_precoder_user_distance_sweep import test_mrc_precoder_user_distance_sweep
from src.analysis.helpers.test_robust_slnr_precoder_error_sweep import test_robust_slnr_precoder_error_sweep
from src.analysis.helpers.test_sac_precoder_error_sweep import test_sac_precoder_error_sweep
from src.analysis.helpers.test_sac_precoder_user_distance_sweep import test_sac_precoder_user_distance_sweep
from src.config.config import Config


if __name__ == '__main__':

    cfg = Config()
    distance_sweep_range = np.arange(75_000, 125_000, 5_000)  # 75km to 125km in 5km steps
    error_sweep_range = np.arange(0.0, 0.2, 0.02)  # 0.0 to 0.2 in 0.02 steps
    
    model_path = Path(
                cfg.trained_models_path,
                '1_sat_16_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading',
                'single_error',
                'userwiggle_50000_snap_4.565',
                'model',
            )
    
    test_mmse_precoder_error_sweep(
        config=cfg,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=100,
    )

    test_mmse_precoder_user_distance_sweep(
        config=cfg,
        distance_sweep_range=distance_sweep_range,
    )

    test_mrc_precoder_error_sweep(
        config=cfg,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=100,
    )

    test_mrc_precoder_user_distance_sweep(
        config=cfg,
        distance_sweep_range=distance_sweep_range,
    )

    test_robust_slnr_precoder_error_sweep(
        config=cfg,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=100,
    )

    test_robust_slnr_precoder_distance_sweep(
        config=cfg,
        distance_sweep_range=distance_sweep_range,
    )

    test_sac_precoder_error_sweep(
        config=cfg,
        model_path=model_path,
        error_sweep_parameter='additive_error_on_cosine_of_aod',
        error_sweep_range=error_sweep_range,
        monte_carlo_iterations=100,
    )

    test_sac_precoder_user_distance_sweep(
        config=cfg,
        distance_sweep_range=distance_sweep_range,
        model_path=model_path,
    )
    
    
    
    
    
    

