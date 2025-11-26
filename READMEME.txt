# Causal AutoML for Zero-Shot Policy Synthesis

This repository contains the implementation of a zero-shot policy synthesis pipeline using causal discovery and resource-aware AutoML.

## Structure
- `environment.py`: Synthetic causal environments
- `causal_discovery.py`: Causal structure discovery module
- `policy_synthesizer.py`: Zero-shot policy generator
- `automl_controller.py`: AutoML module selecting best policy under resource constraints
- `run_demo.py`: End-to-end demonstration
- `utils.py`: Optional helper functions

## Usage
1. Install dependencies: `numpy`
2. Run demo: `python run_demo.py`
