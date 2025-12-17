# Testing Documentation

This directory contains test suites for validating the IRL-based intention recognition framework, including environment functionality tests, gameplay demonstrations, and integration tests.

## 🎯 Overview

The testing infrastructure ensures:

- ✅ Game environments work correctly
- ✅ RL/IRL algorithms train successfully
- ✅ Agent interactions are valid
- ✅ Full pipeline functions end-to-end
- ✅ Models can be loaded and evaluated

---

## 📁 Directory Structure

```
tests/
├── README.md                           # This file
├── test_classic_gym_game_set_up.py    # Basic Gym environment tests
├── test_full_framework_functionality.py # End-to-end integration tests
├── manual_gameplay/                    # Manual gameplay testing
│   ├── play_different_envs_full_obs.py
│   ├── play_different_envs.py
│   ├── play_main_experiment_fixed_horizon_3000.py
│   ├── play_separate_policies.py
│   ├── play_single_policy.py
│   └── example_models/                 # Pre-trained models for testing
│       ├── different_envs/
│       ├── different_envs_full_obs/
│       ├── main_experiment_fixed_horizon_3000/
│       ├── separate_policies/
│       └── single_policy/
└── environment_tests/                          # Environment-specific tests
    ├── test_fixed_horizon_kaz_works.py
    └── test_variable_horizon_kaz_works.py
```

---

## 🧪 Test Categories

### 1. Unit Tests

#### `test_classic_gym_game_set_up.py`

**Purpose**: Validate basic OpenAI Gym environment setup.

**Tests:**

- Environment registration
- Observation/action space correctness
- Reset functionality
- Step function behavior
- Rendering capabilities

**Run:**

```bash
cd tests
python3 test_classic_gym_game_set_up.py
```

**Example Test Cases:**

```python
def test_environment_creation():
    """Test that environments can be created"""
    env = gym.make('kaz-single-player-v0', type_of_player='knight')
    assert env is not None

def test_observation_space():
    """Test observation space is correct shape"""
    env = gym.make('kaz-single-player-v0', type_of_player='archer')
    obs = env.reset()
    assert obs.shape == env.observation_space.shape

def test_action_space():
    """Test action space is discrete and valid"""
    env = gym.make('kaz-double-player-v0', owner_policy=policy)
    assert isinstance(env.action_space, gym.spaces.Discrete)
```

---

### 2. Integration Tests

#### `test_full_framework_functionality.py`

**Purpose**: End-to-end testing of the complete training pipeline.

**Tests:**

- Owner agent training
- Assistant agent training (all stages)
- IRL algorithm integration
- Model saving/loading
- Experiment configuration parsing
- Weights & Biases logging

**Run:**

```bash
cd tests
python3 test_full_framework_functionality.py
```

**Example Test Flow:**

```python
def test_complete_pipeline():
    """Test entire training and evaluation pipeline"""

    # Step 1: Train owner
    owner = Owner()
    owner.train_rl(owner_env_obj, policy='PPO')
    assert owner.rl_policy is not None

    # Step 2: Train assistant RL
    assistant = Assistant()
    assistant.train_rl(assistant_env_obj, policy='PPO')
    assert assistant.rl_policy is not None

    # Step 3: Train IRL
    demos = collect_demonstrations(owner.rl_policy, env)
    assistant.train_irl(demos, irl_algorithm='AIRL')
    assert assistant.irl_model is not None

    # Step 4: Train combiner
    assistant.train_combiner(coop_env_obj, owner.rl_policy)
    assert assistant.action_combiner is not None

    # Step 5: Evaluate
    mean_reward = evaluate_policy(assistant, owner, env)
    assert mean_reward > baseline_reward
```

---

### 3. Environment Tests

Located in `environment_tests/` subdirectory.

#### `test_fixed_horizon_kaz_works.py`

**Purpose**: Validate fixed-horizon KAZ variants.

**Tests:**

- Episode terminates at correct timestep
- Horizon limit is enforced
- Both single and double-player modes
- Different skill levels

**Run:**

```bash
cd tests/environment_tests
python3 test_fixed_horizon_kaz_works.py
```

**Key Validations:**

```python
def test_fixed_horizon_termination():
    """Ensure episodes end at horizon limit"""
    env = gym.make('kaz-fixed-single-player-v0',
                   horizon_limit=1000,
                   type_of_player='knight')

    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

    assert done == True, "Episode should terminate at horizon"

def test_fixed_horizon_double_player():
    """Test fixed horizon with owner policy"""
    owner_policy = PPO.load('example_models/owner.zip')
    env = gym.make('kaz-fixed-double-player-v0',
                   horizon_limit=3000,
                   owner_policy=owner_policy)

    obs = env.reset()
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        steps += 1

    assert steps == 3000, f"Expected 3000 steps, got {steps}"
```

#### `test_variable_horizon_kaz_works.py`

**Purpose**: Validate variable-horizon KAZ variants.

**Tests:**

- Natural termination conditions (death, zombie reaches end)
- Episode length varies appropriately
- Reward structure is correct
- State transitions are valid

**Run:**

```bash
cd tests/environment_tests
python3 test_variable_horizon_kaz_works.py
```

---

### 4. Manual Gameplay Tests

Located in `manual_gameplay/` subdirectory. These allow visual inspection and interactive testing.

#### `play_different_envs.py`

**Purpose**: Visually test different environment variants with local observations.

**Environments Tested:**

- Single-player (knight and archer separately)
- Double-player (cooperative)
- Random baselines
- Single policy variants

**Run:**

```bash
cd tests/manual_gameplay
python3 play_different_envs.py
```

**Features:**

- Renders game in Pygame window
- Uses pre-trained models from `example_models/different_envs/`
- Displays performance metrics
- Records episode statistics

#### `play_different_envs_full_obs.py`

**Purpose**: Same as above but with global observations.

**Use Case**: Validate full observability variants work correctly.

**Run:**

```bash
cd tests/manual_gameplay
python3 play_different_envs_full_obs.py
```

#### `play_main_experiment_fixed_horizon_3000.py`

**Purpose**: Test main experiment configuration with fixed 3000-step episodes.

**Use Case**: Validate ablation study environment settings.

**Run:**

```bash
cd tests/manual_gameplay
python3 play_main_experiment_fixed_horizon_3000.py
```

**Uses Models From:** `example_models/main_experiment_fixed_horizon_3000/`

#### `play_separate_policies.py`

**Purpose**: Test cooperative play with independently trained policies.

**Scenario:**

- Owner trained independently
- Assistant trained independently
- Both play together without coordination training

**Use Case**: Baseline comparison for IRL-based approach.

**Run:**

```bash
cd tests/manual_gameplay
python3 play_separate_policies.py
```

#### `play_single_policy.py`

**Purpose**: Test single policy controlling both agents.

**Scenario:**

- One RL policy outputs actions for both knight and archer
- Maximum coordination (theoretical upper bound)

**Run:**

```bash
cd tests/manual_gameplay
python3 play_single_policy.py
```

---

## 📦 Example Models

Pre-trained models for testing purposes stored in `manual_gameplay/example_models/`.

### Directory Structure

```
example_models/
├── different_envs/
│   ├── knight_policy.zip          # Knight trained in single-player
│   ├── archer_policy.zip          # Archer trained in single-player
│   └── cooperative_policy.zip     # Double-player trained
├── different_envs_full_obs/
│   ├── knight_full_obs.zip
│   └── archer_full_obs.zip
├── main_experiment_fixed_horizon_3000/
│   ├── owner_fixed_3000.zip
│   ├── assistant_rl_fixed_3000.zip
│   └── assistant_complete_fixed_3000.zip
├── separate_policies/
│   ├── owner_independent.zip
│   └── assistant_independent.zip
└── single_policy/
    └── unified_policy.zip
```

### Loading Models

```python
from stable_baselines3 import PPO

# Load owner policy
owner_policy = PPO.load('example_models/separate_policies/owner_independent.zip')

# Load assistant policy
assistant_policy = PPO.load('example_models/separate_policies/assistant_independent.zip')

# Use in environment
env = gym.make('kaz-double-player-v0', owner_policy=owner_policy)
obs = env.reset()
action, _states = assistant_policy.predict(obs)
```

---

## 🚀 Running All Tests

### Quick Test Suite

```bash
cd tests

# Run basic environment tests
python3 test_classic_gym_game_set_up.py

# Run environment-specific tests
python3 environment_tests/test_fixed_horizon_kaz_works.py
python3 environment_tests/test_variable_horizon_kaz_works.py

# Run integration tests
python3 test_full_framework_functionality.py
```

### Visual Validation

```bash
cd tests/manual_gameplay

# Test all environment variants
python3 play_different_envs.py

# Test full observability
python3 play_different_envs_full_obs.py

# Test experiment configurations
python3 play_main_experiment_fixed_horizon_3000.py
python3 play_separate_policies.py
python3 play_single_policy.py
```

---

## 🐛 Common Test Failures

### Environment Creation Errors

**Symptom:** `gym.error.UnregisteredEnv`

**Fix:**

```python
# Ensure environments are registered before tests
from code.src.common.common import register_environments
register_environments()
```

### Model Loading Errors

**Symptom:** `FileNotFoundError` when loading policies

**Fix:**

- Ensure you're in correct directory
- Check model paths are absolute or relative to test script
- Verify example models exist

### CUDA/GPU Errors

**Symptom:** `RuntimeError: CUDA out of memory`

**Fix:**

```python
# Force CPU usage in tests
import torch
device = torch.device('cpu')

model = PPO('MlpPolicy', env, device=device)
```

### Rendering Errors

**Symptom:** Pygame window won't open or crashes

**Fix:**

```bash
# Ensure Pygame is installed
pip install pygame

# On headless systems, use rgb_array mode
env = gym.make('kaz-single-player-v0', render_mode='rgb_array')
```

---

## 📊 Test Coverage

Current test coverage areas:

| Component              | Coverage  | Notes                            |
| ---------------------- | --------- | -------------------------------- |
| Environment Creation   | ✅ High   | All variants tested              |
| Single-player Training | ✅ High   | Both agents tested               |
| Double-player Training | ✅ High   | With owner policy                |
| IRL Training           | ⚠️ Medium | AIRL/GAIL tested, others partial |
| Model Persistence      | ✅ High   | Save/load tested                 |
| Evaluation             | ✅ High   | Multiple scenarios               |
| Edge Cases             | ⚠️ Medium | Some scenarios untested          |

### Areas for Improvement

- [ ] Add more IRL algorithm tests
- [ ] Test failure recovery scenarios
- [ ] Add performance regression tests
- [ ] Test with corrupted models
- [ ] Add multi-seed consistency tests

---

## 🔍 Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In SB3
model = PPO('MlpPolicy', env, verbose=2)
```

### Check Environment

```python
from stable_baselines3.common.env_checker import check_env

env = gym.make('kaz-single-player-v0', type_of_player='knight')
check_env(env)  # Validates Gym interface compliance
```

### Visualize Observations

```python
import matplotlib.pyplot as plt

obs = env.reset()
plt.imshow(obs)
plt.title("Environment Observation")
plt.show()
```

### Profile Training Time

```python
import time

start = time.time()
model.learn(total_timesteps=10000)
elapsed = time.time() - start
print(f"Training took {elapsed:.2f} seconds")
```

---

## 📚 Best Practices

### 1. Test Incrementally

- Test small components before integration
- Use example models for quick validation
- Start with simple environments

### 2. Reproducibility

```python
import random
import numpy as np
import torch

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seeds()  # Call before tests
```

### 3. Isolate Tests

- Each test should be independent
- Clean up environments after tests
- Don't rely on test execution order

### 4. Document Expected Behavior

```python
def test_episode_reward():
    """
    Test that episode rewards are within expected range.

    Expected: Rewards should be between -100 and 1000
    for a well-trained policy in vanilla difficulty.
    """
    # Test implementation
```

---

## 🛠️ Creating New Tests

### Template for Unit Test

```python
import gymnasium as gym
import sys
sys.path.append('../code/src')

from common.common import register_environments

def test_my_feature():
    """Test description"""
    register_environments()

    # Setup
    env = gym.make('kaz-single-player-v0', type_of_player='knight')

    # Execute
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    # Assert
    assert obs is not None, "Observation should not be None"
    assert isinstance(reward, (int, float)), "Reward should be numeric"

    # Cleanup
    env.close()
    print("✓ test_my_feature passed")

if __name__ == '__main__':
    test_my_feature()
```

### Template for Integration Test

```python
from agents.owner import Owner
from agents.assistant import Assistant
from agents.helpers.env_config import EnvObject

def test_training_pipeline():
    """Test complete training flow"""

    # Train owner
    owner = Owner()
    owner_env = EnvObject(
        env_id='kaz-single-player-v0',
        n_envs=2,
        total_timesteps=10000,  # Small for testing
        env_kwargs={'type_of_player': 'knight'}
    )
    owner.train_rl(owner_env, policy='PPO')

    # Train assistant
    assistant = Assistant()
    assistant_env = EnvObject(
        env_id='kaz-single-player-v0',
        n_envs=2,
        total_timesteps=10000,
        env_kwargs={'type_of_player': 'archer'}
    )
    assistant.train_rl(assistant_env, policy='PPO')

    # Validate
    assert owner.rl_policy is not None
    assert assistant.rl_policy is not None

    print("✓ test_training_pipeline passed")
```

---

## 📞 Support

If tests fail unexpectedly:

1. Check environment registration
2. Verify model paths
3. Ensure dependencies are installed
4. Check GPU/CUDA availability
5. Review error logs carefully

For persistent issues, review the main documentation or experiment configurations.

---

This testing infrastructure ensures the reliability and correctness of the IRL-based intention recognition framework throughout development and experimentation.
