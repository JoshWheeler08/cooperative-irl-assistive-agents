# Quick Reference Guide

A concise reference for common tasks and commands in the IRL-based Intention Recognition framework.

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/JoshWheeler08/Investigating-IRL-based-intention-recognition-algorithms.git
cd Investigating-IRL-based-intention-recognition-algorithms
python3 -m venv env && source env/bin/activate
pip install -r code/requirements.txt

# Run experiments
cd code/src
python3 experiments/run_baselines.py        # Baseline experiments
python3 experiments/run_main.py              # IRL experiments
```

---

## 📂 Project Structure at a Glance

```
Repository Root/
├── README.md                  # Main documentation
├── SETUP.md                   # Installation guide
├── CONTRIBUTING.md            # Contribution guide
├── LICENSE                    # MIT License
├── CITATION.cff              # Citation information
├── code/
│   ├── requirements.txt      # Python dependencies
│   └── src/
│       ├── run_main_experiment.py        # Main IRL experiments
│       ├── run_kaz_baselines.py          # Baseline tests
│       ├── configuration/                # YAML configs
│       ├── irl_training/                 # IRL algorithms
│       ├── envs/                         # Game environments
│       └── MyAgents/                     # Agent classes
├── tests/                     # Test suites
└── docker/                   # Docker setup
```

---

## 🎮 Running Experiments

### Baseline Experiments

```bash
cd code/src
python3 experiments/run_baselines.py [config_file]
# Default: configuration/kaz_baselines_config.yaml
```

**What it tests:**

- Random actions
- Single trained policy
- Separate policies
- Various cooperation levels

### Main IRL Experiments

```bash
cd code/src
python3 experiments/run_main.py [config_file]
# Default: configuration/main_experiment_config.yaml
```

**What it trains:**

1. Owner agent (human player)
2. Assistant RL policy
3. Assistant IRL model
4. Combined assistant

---

## ⚙️ Configuration Cheat Sheet

### Key Configuration Parameters

```yaml
# In configuration/*.yaml files

experiment:
  n_envs: 4 # Parallel environments (↑ = faster training)
  total_timesteps: 1M # Training duration (↑ = better convergence)
  seeds: [42, 123] # Random seeds for reproducibility

owner_agent:
  rl_algorithm: "PPO" # Options: PPO, A2C, DQN
  policy_type: "MlpPolicy"

assistant_agent:
  rl_algorithm: "PPO"
  irl_algorithm: "AIRL" # Options: AIRL, GAIL, BC, DAgger

environment:
  env_id: "kaz-double-player-v0"
  skill_level: "medium" # Options: vanilla, medium, hard
  horizon: "variable" # Options: variable, fixed
  horizon_limit: 3000 # Only if horizon: fixed
```

---

## 🤖 IRL Algorithms Quick Reference

| Algorithm       | Type                 | Best For                | Speed  |
| --------------- | -------------------- | ----------------------- | ------ |
| **AIRL**        | Reward learning      | Intention understanding | Slow   |
| **GAIL**        | Direct imitation     | Behavior matching       | Medium |
| **BC**          | Supervised learning  | Quick baseline          | Fast   |
| **DAgger**      | Interactive learning | Iterative improvement   | Medium |
| **Density**     | Density estimation   | Fast approximation      | Fast   |
| **Pref. Comp.** | Preference-based     | Human feedback          | Slow   |

### Algorithm Selection

```python
# In configuration file
assistant_agent:
  irl_algorithm: "AIRL"  # Choose one
```

---

## 🎯 Environment Variants

### Single-Player Environments

```python
# Train one agent in isolation
env = gym.make('kaz-single-player-v0', type_of_player='knight')
env = gym.make('kaz-single-player-v0', type_of_player='archer')
```

### Double-Player Environments

```python
# Cooperative training (requires owner policy)
env = gym.make('kaz-double-player-v0', owner_policy=policy)
```

### Variants

| Suffix            | Description               |
| ----------------- | ------------------------- |
| `fixed-horizon-*` | Fixed episode length      |
| `full-obs-*`      | Global observations       |
| `random-*`        | Random actions (baseline) |
| `single-policy-*` | One policy controls both  |

---

## 🧪 Testing Commands

```bash
cd tests

# Unit tests
python3 test_classic_gym_game_set_up.py

# Environment tests
python3 environment_tests/test_fixed_horizon_kaz_works.py
python3 environment_tests/test_variable_horizon_kaz_works.py

# Integration tests
python3 test_full_framework_functionality.py

# Visual tests
cd manual_gameplay
python3 play_different_envs.py
```

---

## 🐛 Common Issues & Quick Fixes

### Issue: Module not found

```bash
source env/bin/activate
pip install -r requirements.txt
```

### Issue: Unregistered environment

```python
from common.common import register_environments
register_environments()
```

### Issue: CUDA out of memory

```yaml
# In config file, reduce:
n_envs: 2 # From 4 or 8
```

### Issue: Slow training

```yaml
n_envs: 8 # Increase parallel envs
# Or use GPU if available
```

### Issue: W&B errors

```bash
export WANDB_MODE=disabled  # Disable W&B
# Or login: wandb login
```

---

## 📊 Model Files

### Saving Models

```python
# Models auto-saved by experiments
# Location: code/src/main_experiment_output/
```

### Loading Models

```python
from stable_baselines3 import PPO

model = PPO.load("path/to/model.zip")
action, _states = model.predict(observation)
```

---

## 🔧 Development Workflow

### 1. Setup

```bash
git clone <repo>
cd <repo>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Create Branch

```bash
git checkout -b feature/my-feature
```

### 3. Make Changes

```bash
# Edit code
# Add tests
# Update docs
```

### 4. Test

```bash
cd tests
python3 test_*.py
```

### 5. Commit

```bash
git add .
git commit -m "[Feature] Description"
git push origin feature/my-feature
```

### 6. Pull Request

Create PR on GitHub with description

---

## 📦 Docker Quick Commands

```bash
cd docker

# Build
docker build -t irl-intention-recognition .

# Run with GPU
docker run --gpus all -v $(pwd):/workspace -it irl-intention-recognition

# Run without GPU
docker run -v $(pwd):/workspace -it irl-intention-recognition

# Inside container
cd /workspace/code/src
python3 run_main_experiment.py
```

---

## 📈 Weights & Biases Integration

```bash
# Setup (one-time)
pip install wandb
wandb login

# Use in code (automatic in experiments)
import wandb
wandb.init(project="my-project")
wandb.log({"metric": value})

# Disable if needed
export WANDB_MODE=disabled
```

---

## 🔍 File Locations

| What                  | Where                              |
| --------------------- | ---------------------------------- |
| **Main README**       | `README.md`                        |
| **Setup Guide**       | `SETUP.md`                         |
| **Contributing**      | `CONTRIBUTING.md`                  |
| **Configurations**    | `code/src/configuration/`          |
| **IRL Algorithms**    | `code/src/irl_training/`           |
| **Environments**      | `code/src/environments/`           |
| **Tests**             | `tests/`                           |
| **Experiment Output** | `code/src/main_experiment_output/` |
| **Docker Files**      | `docker/`                          |

---

## 🎓 Key Papers & Resources

### IRL Algorithms

- **AIRL**: Fu et al. (2017)
- **GAIL**: Ho & Ermon (2016)
- **DAgger**: Ross et al. (2011)

### Libraries

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Imitation**: https://imitation.readthedocs.io/
- **PettingZoo**: https://pettingzoo.farama.org/

---

## 💡 Pro Tips

1. **Start Small**: Use small `total_timesteps` for testing
2. **Use W&B**: Track experiments for easy comparison
3. **Test Incrementally**: Test components before full runs
4. **Save Often**: Models auto-save but check paths
5. **Monitor GPU**: Use `nvidia-smi` to check utilization
6. **Parallel Envs**: Increase `n_envs` for faster training
7. **Multiple Seeds**: Run with different seeds for robustness
8. **Document Changes**: Update configs and docs

---

## 🆘 Getting Help

- **Docs**: Read `README.md`, `SETUP.md`, `CONTRIBUTING.md`
- **Issues**: Search [GitHub Issues](https://github.com/JoshWheeler08/cooperative-irl-assistive-agents/issues)
- **New Issue**: Open with detailed description
- **Community**: GitHub Discussions

---

## 📝 Citation

```bibtex
@mastersthesis{wheeler2023irl,
  title={Investigating Human Intention Recognition Algorithms based on Inverse Reinforcement Learning for Assistive Action Planning},
  author={Wheeler, Joshua},
  year={2023},
  school={University of St Andrews}
}
```

---

**Keep this guide handy for quick reference!** 📌

For detailed information, see the full documentation in `README.md` and other guides.
