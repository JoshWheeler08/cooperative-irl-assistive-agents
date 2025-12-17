# Source Code Documentation

This directory contains the core implementation of the IRL-based intention recognition framework for assistive action planning.

## 📂 Directory Structure

### Core Experiment Files

#### `experiment_core.py`

The heart of the experiment system. Contains the core orchestration logic for:

- Training the owner/human agent using deep RL
- Training the assistant agent with IRL + deep RL
- Evaluating cooperative performance
- Logging results to Weights & Biases

**Key Functions:**

- `_train_owner_agent()` - Trains the human/owner agent in single-player mode
- `_train_assistant_agent()` - Trains the IRL-based assistant agent
- `run_experiment()` - Main experiment orchestration function

#### `run_main.py`

Command-line interface for running IRL-based experiments. Reads configuration from YAML files and executes the full experimental pipeline.

**Usage:**

```bash
python3 run_main.py [config_file_path]
# Default: configuration/main_experiment_config.yaml
```

#### `run_baselines.py`

Executes baseline experiments that establish performance bounds without IRL-based assistance.

**Usage:**

```bash
python3 run_baselines.py [config_file_path]
# Default: configuration/kaz_baselines_config.yaml
```

#### Shell Scripts

- `run_main_tests.sh` - Convenient wrapper for main experiments
- `run_baseline_tests.sh` - Convenient wrapper for baseline experiments

---

## 📁 Module Descriptions

### `baselines/`

Baseline experiment implementations for comparison with IRL-based approaches.

**`kaz/kaz_baselines.py`** - Six baseline tests with varying cooperation levels:

1. Random actions (both agents)
2. Random shared policy
3. Single trained policy controlling both agents
4. Knight-only trained
5. Archer-only trained
6. Separately trained policies

**`kaz/example_results/`** - Sample output data from baseline experiments

---

### `common/`

Shared utilities used across experiments.

**`common.py`** - Core utility functions:

- `setup_wandb()` - Initialize experiment tracking
- `log_results_to_wandb()` - Log metrics to W&B
- `create_tables_from_results()` - Generate result summaries
- `get_random_seed()` - Reproducible random seed generation
- `register_environments()` - Register custom Gym environments

**`constants.py`** - Shared constants and configuration values:

- Environment IDs
- Default hyperparameters
- Path configurations

---

### `configuration/`

YAML configuration files for experiment hyperparameters.

**`kaz_baselines_config.yaml`** - Baseline experiment settings:

- Number of training environments
- Total timesteps
- RL algorithm selection (PPO, A2C, DQN)
- Random seeds for reproducibility
- Environment variants (horizon, skill level)

**`main_experiment_config.yaml`** - Main IRL experiment settings:

- All baseline config options PLUS:
- IRL algorithm selection (AIRL, GAIL, BC, DAgger, etc.)
- IRL-specific hyperparameters
- Assistant agent training parameters
- Action combiner (MLP) configuration

**Example Configuration Structure:**

```yaml
experiment:
  name: "IRL_AIRL_PPO_Experiment"
  n_envs: 4
  total_timesteps: 1000000

owner_agent:
  rl_algorithm: "PPO"
  policy_type: "MlpPolicy"

assistant_agent:
  rl_algorithm: "PPO"
  irl_algorithm: "AIRL"

environment:
  skill_level: "medium"
  horizon: "variable"
```

---

### `environments/`

Game environment implementations. See [environments/README.md](environments/README.md) for detailed documentation.

**`kaz_core/`** - Vanilla KAZ implementations:

- `single_player.py` - One agent training
- `double_player.py` - Two agent training with one policy provided
- `fixed_horizon_single_player.py` - Fixed episode length variant
- `fixed_horizon_double_player.py` - Fixed episode length for two players

**`kaz_variants/`** - Modified KAZ environments:

- `fixed_horizon/` - Fixed-horizon variants with modified observations/actions
- `variable_horizon/` - Variable-horizon variants
- `eval/` - Evaluation-specific environment wrappers

**Key Environment Modifications:**

- Full observability variants (global vs local observation)
- Random action variants (for baseline testing)
- Single policy variants (one policy controls both agents)
- Skill level variations (arrow speed adjustments)

---

### `irl_training/`

IRL and imitation learning algorithm implementations. See [irl_training/README.md](irl_training/README.md) for details.

**Implemented Algorithms:**

- **`airl.py`** - Adversarial Inverse Reinforcement Learning
- **`gail.py`** - Generative Adversarial Imitation Learning
- **`behavioral_cloning.py`** - Behavioral Cloning
- **`dagger.py`** - Dataset Aggregation
- **`density_based_reward_modelling.py`** - Kernel density reward estimation
- **`pref_comp.py`** - Preference Comparisons

**`irl_base.py`** - Abstract base class providing standard interface for all IRL implementations.

**Standard Interface Methods:**

- `train()` - Train the IRL algorithm on expert demonstrations
- `get_reward()` - Extract learned reward function
- `get_policy()` - Get the learned policy

---

### `agents/`

Agent abstractions for owner and assistant. See [agents/README.md](agents/README.md) for details.

**Core Classes:**

- **`Agent.py`** - Abstract base class with shared methods
- **`Owner.py`** - Human/owner agent implementation
- **`Assistant.py`** - IRL-based assistant agent implementation

**`Objects/`** - Data encapsulation classes:

- **`EnvObject.py`** - Environment configuration and state
- **`PolicyObject.py`** - Policy configuration and model storage

**Key Methods:**

- `train_rl()` - Train agent using reinforcement learning
- `train_irl()` - Train using inverse reinforcement learning
- `predict()` - Generate action predictions
- `save()` / `load()` - Model persistence

---

### `wrappers/`

OpenAI Gym wrappers for environment behavior modification.

**`kaz_training_wrapper.py`** - Custom wrapper that:

- Modifies observation spaces
- Adds reward shaping
- Implements episode termination logic
- Supports both single and multi-agent training

**Usage:**

```python
from Wrappers.KAZTrainingWrapper import KAZTrainingWrapper
wrapped_env = KAZTrainingWrapper(base_env, **wrapper_kwargs)
```

---

### `main_experiment_output/`

Experiment results and artifacts.

**`official/`** - Official experiment results

- `variable/` - Variable horizon results
- `fixed/` - Fixed horizon results
- Trained models (`.zip` files)
- Performance logs
- Weights & Biases sync data

**Structure:**

```
main_experiment_output/
└── official/
    └── variable/
        ├── experiment_1/
        │   ├── owner_model.zip
        │   ├── assistant_model.zip
        │   ├── results.json
        │   └── wandb/
        └── experiment_2/
            └── ...
```

---

### `licences_of_libraries_used/`

License files for third-party libraries used in this project:

- `imitationLicense` - Imitation library
- `OpenAIGymLicense` - OpenAI Gym
- `PZ-License` - PettingZoo
- `StableBaselines3License` - Stable-Baselines3

---

## 🔧 Configuration Guide

### Modifying Experiments

1. **Change RL Algorithm:**

   ```yaml
   owner_agent:
     rl_algorithm: "PPO" # Options: PPO, A2C, DQN
   ```

2. **Change IRL Algorithm:**

   ```yaml
   assistant_agent:
     irl_algorithm: "AIRL" # Options: AIRL, GAIL, BC, DAgger, etc.
   ```

3. **Adjust Training Duration:**

   ```yaml
   experiment:
     total_timesteps: 1000000 # Increase for better convergence
   ```

4. **Modify Environment Difficulty:**

   ```yaml
   environment:
     skill_level: "medium" # Options: vanilla, medium, hard
   ```

5. **Fixed vs Variable Horizon:**

   ```yaml
   environment:
     horizon: "variable" # Options: variable, fixed
     horizon_limit: 3000 # Only used if horizon: fixed
   ```

   **⚠️ Important Research Finding:**

   - **Fixed horizons** (3000 timesteps) produced significantly better IRL performance
   - **Variable horizons** showed mixed results, possibly due to imitation library optimization
   - Recommendation: Start with fixed horizons for IRL experiments

---

## 🚀 Extending the Framework

### Adding a New IRL Algorithm

1. Create a new file in `irl_training/`
2. Inherit from `IRLTemplate` class
3. Implement required methods: `train()`, `get_reward()`, `get_policy()`
4. Update configuration to include new algorithm option

### Adding a New Game Environment

1. Create OpenAI Gym compatible environment
2. Ensure it supports the required interface:
   - Two-player cooperative gameplay
   - Distinct action spaces for each agent
   - Appropriate observation spaces
3. Register environment in `common/common.py`
4. Update configuration files

### Custom Metrics

Add custom logging in `experiment_core.py`:

```python
wandb.log({
    "custom_metric": value,
    "step": timestep
})
```

---

## 📊 Understanding the Output

### Console Output

- Training progress bars (via tqdm)
- Episode rewards and lengths
- IRL training metrics
- Model save confirmations

### Weights & Biases Dashboard

- Real-time training curves
- Hyperparameter tracking
- Model versioning
- Comparative analysis across runs

### Saved Models

- Located in `main_experiment_output/`
- `.zip` format (Stable-Baselines3 standard)
- Load with: `model = PPO.load("path/to/model.zip")`

---

## 🐛 Troubleshooting

**CUDA Out of Memory:**

- Reduce `n_envs` in configuration
- Use smaller neural network architectures
- Train on CPU (slower but more reliable)

**Slow Training:**

- Increase `n_envs` for parallel environment simulation
- Use GPU acceleration
- Reduce `total_timesteps` for faster iterations

**Poor Performance:**

- Increase `total_timesteps`
- Tune RL hyperparameters (learning rate, batch size)
- Try different IRL algorithms
- Check expert demonstration quality

---

## 📚 Additional Resources

- **Stable-Baselines3 Docs:** https://stable-baselines3.readthedocs.io/
- **Imitation Library Docs:** https://imitation.readthedocs.io/
- **PettingZoo Docs:** https://pettingzoo.farama.org/
- **W&B Documentation:** https://docs.wandb.ai/
