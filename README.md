# Satellite Communications Final Project

This repository contains the implementation and analysis code for the paper "Deep Learning-Based Beamforming for Satellite Communications" (2402.16563v1). The project implements a Soft Actor-Critic (SAC) reinforcement learning algorithm for optimizing beamforming in satellite communications systems.

## Project Structure

- `src/`: Main source code directory
  - `models/`: Implementation of the SAC algorithm and training code
  - `analysis/`: Scripts for analyzing model performance and generating results
  - `plotting/`: Visualization tools for results and beam patterns
  - `utils/`: Utility functions and helper modules
  - `data/`: Data processing and management
  - `config/`: Configuration files
  - `tests/`: Test cases
- `paper/`: Contains the research paper (2402.16563v1.pdf)
- `reports/`: Generated analysis reports and results

## Prerequisites

- Python 3.8 or higher
- All required Python packages listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd satellite-communications_final_project
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Recreating Paper Results

### 1. Training the Model

The main training script is located at `src/models/train_sac.py`. To train the model:

```bash
python src/models/train_sac.py
```

For hyperparameter sweeps:
```bash
python src/models/train_sac_sweep.py
```

### 2. Generating Beam Patterns

After training, generate beam patterns using:
```bash
python src/analysis/generate_beampatterns.py
```

### 3. Running Analysis

To analyze model performance:
```bash
# Test beam pattern performance
python src/analysis/test_beam_pattern.py

# Run time analysis
python src/analysis/run_time_analysis.py
```

### 4. Generating Plots

Visualize the results using the plotting scripts:
```bash
# Plot training progress
python src/plotting/plot_training_graph.py

# Plot error sweep results
python src/plotting/plot_error_sweep_testing_graph.py

# Plot distance sweep results
python src/plotting/plot_distance_sweep_testing_graph.py

# Plot beam patterns
python src/plotting/plot_beam_patterns.py
```

## Key Results Reproduction

The paper presents several key results that can be reproduced using this codebase:

1. **Training Performance**: Use `plot_training_graph.py` to visualize the learning curves
2. **Beam Pattern Analysis**: Use `generate_beampatterns.py` and `plot_beam_patterns.py` to recreate the beam pattern figures
3. **Error Analysis**: Use `test_beam_pattern.py` and `plot_error_sweep_testing_graph.py` for error analysis
4. **Distance Analysis**: Use `plot_distance_sweep_testing_graph.py` for distance-based performance analysis

## Configuration

The main configuration parameters can be found in:
- `src/config/`: Contains configuration files for different aspects of the system
- Training parameters can be modified in `src/models/train_sac.py`

## Notes

- Training times may vary depending on your hardware
- The default hyperparameters are set to match those used in the paper
- For best results, use a GPU for training
- Some analysis scripts may require significant computational resources

## Citation

If you use this code in your research, please cite the original paper:
```
[Paper citation will be added here]
```