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

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Paper Results

Follow these steps in order to reproduce the paper results:

### 1. Training the Model

The main training script is located at `src/models/train_sac.py`. To train the model:

```bash
python src/models/train_sac.py
```

This script will:
- Train the SAC model with the optimal hyperparameters
- Save the trained model in the `results/models` directory
- Generate initial performance metrics

### 2. Generating Beam Patterns

After training completes, generate the beam patterns:

```bash
python src/analysis/generate_beampatterns.py
```

This will create beam pattern visualizations in the `results\metrics` directory.

### 3. Running Performance Analysis

To analyze the model's performance under different conditions:

```bash
# Test beam pattern performance
python src/analysis/test_beam_pattern.py

# Run time analysis
python src/analysis/run_time_analysis.py
```

### 4. Generating Final Plots

The final step is to generate the paper's figures:

```bash
# Plot training progress
python src/plotting/plot_training_graph.py

# Plot error sweep results
python src/plotting/plot_error_sweep_testing_graph.py

# Plot beam patterns
python src/plotting/plot_beam_patterns.py
```

## Expected Results

After completing all steps, you should have:

1. Trained model files in `models/`
2. Beam pattern visualizations in `results/beam_patterns/`
3. Performance metrics in `results/metrics/`
4. Final plots in `results/plots/`

## Configuration

The training and analysis parameters are configured in:
- `src/config/config.py`: Main configuration file with all parameters

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Check that the virtual environment is activated
3. Verify that the required directories exist in the `results` folder
4. Check the logs in `results/logs/training.log` for any errors
