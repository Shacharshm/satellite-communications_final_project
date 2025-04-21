from pathlib import Path
import numpy as np
from scipy import constants
from tensorflow import get_logger as tf_get_logger
import logging
from logging.handlers import RotatingFileHandler
from sys import stdout

from src.config.config_error_model import ConfigErrorModel
from src.config.config_sac_learner import ConfigSACLearner
from src.data.channel.los_channel_model import los_channel_model
from src.utils.get_wavelength import get_wavelength

class MinimalConfig:
    """Minimal configuration for reproducing paper results."""
    
    def __init__(self) -> None:
        self._pre_init()
        
        # Essential Parameters
        self.freq = 2 * 10**9  # 2 GHz carrier frequency
        self.noise_power_watt = 10**(7/10) * 290 * constants.value('Boltzmann constant') * 30 * 10**6
        self.power_constraint_watt = 100  # Transmit power constraint
        
        # Satellite Parameters
        self.sat_nr = 1  # Single satellite
        self.sat_tot_ant_nr = 16  # Total number of Tx antennas
        self.sat_gain_dBi = 20  # Satellite antenna gain
        self.sat_ant_nr = self.sat_tot_ant_nr // self.sat_nr
        
        # User Parameters
        self.user_nr = 3  # Number of users
        self.user_gain_dBi = 0  # User antenna gain
        self.user_dist_average = 100_000  # Average user distance (m)
        self.user_dist_bound = 50_000  # User distance variation bound
        
        # Channel Model
        self.channel_model = los_channel_model
        self.wavelength = get_wavelength(self.freq)
        
        self._post_init()
    
    def _pre_init(self) -> None:
        self.rng = np.random.default_rng(seed=42)  # Fixed seed for reproducibility
        self.logger = logging.getLogger()
        
        # Essential paths
        self.project_root_path = Path(__file__).parent.parent.parent
        self.output_metrics_path = Path(self.project_root_path, 'results', 'training')
        self.trained_models_path = Path(self.project_root_path, 'results', 'models')
        
        # Create directories
        self.output_metrics_path.mkdir(parents=True, exist_ok=True)
        self.trained_models_path.mkdir(parents=True, exist_ok=True)
    
    def _post_init(self) -> None:
        # Error Model Configuration
        self.config_error_model = ConfigErrorModel(
            channel_model=self.channel_model,
            rng=self.rng,
            wavelength=self.wavelength,
            user_nr=self.user_nr
        )
        
        # SAC Learner Configuration
        self.config_learner = ConfigSACLearner(
            sat_nr=self.sat_nr,
            sat_ant_nr=self.sat_ant_nr,
            user_nr=self.user_nr
        )
        
        # Logging Setup
        self.logfile_path = Path(self.project_root_path, 'results', 'logs', 'training.log')
        self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Essential Arguments
        self.satellite_args = {
            'rng': self.rng,
            'antenna_nr': self.sat_ant_nr,
            'antenna_distance': 3 * self.wavelength / 2,
            'antenna_gain_linear': 10**(self.sat_gain_dBi/10) / self.sat_tot_ant_nr,
            'user_nr': self.user_nr,
            'freq': self.freq,
            'center_aod_earth_deg': 90,
            'error_functions': self.config_error_model.error_rngs
        }
        
        self.user_args = {
            'gain_linear': 10**(self.user_gain_dBi/10),
        }
        
        self.mmse_args = {
            'power_constraint_watt': self.power_constraint_watt,
            'noise_power_watt': self.noise_power_watt,
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
        }
        
        self.learned_precoder_args = {
            'sat_nr': self.sat_nr,
            'sat_ant_nr': self.sat_ant_nr,
            'user_nr': self.user_nr,
            'power_constraint_watt': self.power_constraint_watt,
        }
    
    def _setup_logging(self) -> None:
        formatter = logging.Formatter(
            '{asctime} : {levelname:8s} : {name:30} : {funcName:25} :: {message}',
            datefmt='%Y-%m-%d %H:%M:%S',
            style='{'
        )
        
        file_handler = RotatingFileHandler(self.logfile_path, maxBytes=10_000_000, backupCount=1)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.setLevel(logging.NOTSET)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Configure TensorFlow logging
        tf_logger = tf_get_logger()
        tf_logger.setLevel(logging.WARNING) 