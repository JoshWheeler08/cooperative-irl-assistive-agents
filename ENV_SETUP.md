# Environment Setup Instructions

## Quick Setup

Follow these steps to set up your development environment:

### 1. Create Virtual Environment

```bash
cd /Users/joshuawheeler/PersonalProjects/Investigating-IRL-based-intention-recognition-algorithms
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install Dependencies

**Option A: Install as editable package (recommended for development)**

```bash
pip install --upgrade pip
pip install -e .
```

This installs the package in editable mode, so imports work everywhere and your IDE won't show import errors.

**Option B: Install from requirements.txt**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Note: With this option, the sys.path setup in files is needed for imports to work.

### 3. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your WandB API key
# Get your key from: https://wandb.ai/authorize
nano .env  # or use your preferred editor
```

### 4. Running Experiments

All commands should be run from the `code/src` directory:

```bash
cd code/src

# Run baseline experiments
python3 experiments/run_baselines.py configuration/kaz_baselines_config.yaml

# Run main IRL experiments
python3 experiments/run_main.py configuration/main_experiment_config.yaml

# Or use convenience scripts
bash scripts/run_baseline_tests.sh
bash scripts/run_main_tests.sh
```

### 5. Running Tests

```bash
cd code

# Run framework functionality test
python3 tests/test_full_framework_functionality.py

# Run environment tests
python3 tests/environment_tests/test_variable_horizon_kaz_works.py
python3 tests/environment_tests/test_fixed_horizon_kaz_works.py

# Run manual gameplay tests
python3 tests/manual_gameplay/play_separate_policies.py
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`, ensure:

1. You're in a virtual environment with dependencies installed
2. You're running scripts from the correct directory (`code/src` for experiments)
3. The `sys.path` setup code is present in entry point files

### Missing Modules

If specific modules are missing:

```bash
pip install numpy torch stable-baselines3 imitation gymnasium pettingzoo pygame wandb python-dotenv pyyaml
```

## Project Structure

```
code/src/           # Source code (run experiments from here)
├── agents/         # Agent implementations
├── baselines/      # Baseline experiments
├── common/         # Shared utilities
├── configuration/  # YAML config files
├── environments/   # Game environments
├── experiments/    # Experiment runners (entry points)
├── irl_training/   # IRL algorithms
├── scripts/        # Shell scripts
└── wrappers/       # Gym wrappers

code/tests/         # Test files (run tests from code/ directory)
├── environment_tests/
├── manual_gameplay/
└── *.py           # Test scripts
```

## Notes

- All Python files in `experiments/` and `tests/` have been configured with automatic path setup
- No need to manually set PYTHONPATH
- Dependencies are in the root `requirements.txt`
