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

### 1. Generating Beam Patterns

After training completes, generate the beam patterns:

```bash
python src/analysis/generate_beampatterns.py
```

This will create beam pattern visualizations in the `results\metrics` directory.

### 2. Running Performance Analysis

To analyze the model's performance under different conditions:

```bash
# Test beam pattern performance
python src/analysis/test_beam_pattern.py

# Run time analysis
python src/analysis/run_time_analysis.py

# Run testing
python src/analysis/run_tests.py
```

### 3. Generating Final Plots

The final step is to generate the paper's figures:

```bash
# Plot error sweep results
python src/plotting/plot_error_sweep_testing_graph.py

# Plot distance sweep results
python src/plotting/plot_distance_sweep_testing_graph.py

# Plot beam patterns
python src/plotting/plot_beam_patterns.py
```

## Expected Results

After completing all steps, you should have:

1. Beam pattern visualizations in `results/beam_patterns/`
2. Performance metrics in `results/metrics/`
3. Final plots in `results/plots/`

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Check that the virtual environment is activated
3. Verify that the required directories exist in the `results` folder
4. Check the logs in `results/logs/training.log` for any errors
