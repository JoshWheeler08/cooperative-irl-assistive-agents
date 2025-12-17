# Contributing to IRL-based Intention Recognition Research

Thank you for your interest in contributing to this project! This guide will help you understand how to contribute effectively.

## 🎯 Project Overview

This is a research project investigating Inverse Reinforcement Learning (IRL) algorithms for human intention recognition in assistive systems. The codebase provides a flexible framework for testing different IRL and RL algorithms in cooperative multi-agent environments.

## 📋 Ways to Contribute

### 1. Research Contributions

- Implement new IRL algorithms
- Add support for new game environments
- Improve existing algorithms
- Add new evaluation metrics
- Contribute experimental results

### 2. Code Improvements

- Bug fixes
- Performance optimizations
- Code refactoring
- Documentation improvements
- Test coverage expansion

### 3. Documentation

- Improve README files
- Add tutorials and examples
- Document configuration options
- Create video demonstrations
- Write blog posts about your experiments

## 🚀 Getting Started

### Prerequisites

See [SETUP.md](SETUP.md) for detailed system requirements and installation instructions.

**Quick checklist:**

- Python 3.8+
- Git
- CUDA-capable GPU (recommended)
- Docker (optional, for reproducible environments)

### Development Setup

1. **Fork and Clone**

```bash
git clone https://github.com/YOUR_USERNAME/Investigating-IRL-based-intention-recognition-algorithms.git
cd Investigating-IRL-based-intention-recognition-algorithms
```

2. **Install Dependencies**

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

3. **Configure WandB (Optional)**

```bash
# Set your WandB API key
export WANDB_API_KEY="your_api_key_here"
```

See [SECURITY.md](SECURITY.md) for secure credential management.

4. **Verify Installation**

```bash
cd code/src
python3 -c "from common.common import register_environments; register_environments(); print('Setup successful!')"
```

5. **Create Development Branch**

```bash
git checkout -b feature/your-feature-name
```

## 🏗️ Project Structure

Understanding the architecture is crucial for contributions:

```
code/src/
├── baselines/          # Baseline experiments
├── common/             # Shared utilities
├── configuration/      # YAML configs
├── environments/       # Game environments
├── irl_training/       # IRL algorithms (EXTEND HERE for new algorithms)
├── agents/             # Agent abstractions
├── wrappers/           # Gym wrappers
├── experiments/        # Experiment runners
└── scripts/            # Shell scripts
```

### Key Extension Points

| What to Add       | Where              | What to Extend           |
| ----------------- | ------------------ | ------------------------ |
| New IRL Algorithm | `irl_training/`    | `IRLTemplate`            |
| New Environment   | `envs/`            | Base environment classes |
| New Agent Type    | `MyAgents/`        | `Agent` ABC              |
| New Baseline      | `baselines/`       | Baseline framework       |
| New Metric        | `common/common.py` | Logging functions        |

## 🔧 Adding a New IRL Algorithm

### Step-by-Step Guide

1. **Create Algorithm File**

```bash
cd code/src/irl_training
touch my_new_irl.py
```

2. **Implement Algorithm**

```python
# my_new_irl.py
from irl_training.irl_base import IRLBase
from imitation.algorithms import <base_algorithm>

class MyNewIRL(IRLBase):
    """
    Brief description of algorithm.

    References:
    - Paper citation
    - Algorithm details
    """

    def __init__(self, demonstrations, env, **kwargs):
        """
        Initialize algorithm.

        Args:
            demonstrations: Expert trajectories
            env: Training environment
            **kwargs: Algorithm-specific parameters
        """
        super().__init__()
        self.demonstrations = demonstrations
        self.env = env
        # Initialize your algorithm components

    def train(self, n_epochs=100, **kwargs):
        """
        Train the IRL algorithm.

        Args:
            n_epochs: Number of training epochs
            **kwargs: Additional training parameters
        """
        # Training logic here
        pass

    def get_reward(self):
        """Return learned reward function (if applicable)"""
        return self.reward_function

    def get_policy(self):
        """Return learned/imitated policy"""
        return self.learned_policy

    def save(self, path):
        """Save trained model"""
        # Save logic
        pass

    def load(self, path):
        """Load trained model"""
        # Load logic
        pass
```

3. **Add to Configuration**

```yaml
# configuration/main_experiment_config.yaml
assistant_agent:
  irl_algorithm: "MyNewIRL" # Add your algorithm name
  irl_kwargs:
    # Algorithm-specific hyperparameters
    param1: value1
    param2: value2
```

4. **Update Algorithm Registry**

```python
# common/common.py or create irl_training/__init__.py
def get_irl_algorithm(name):
    algorithms = {
        'AIRL': AIRL,
        'GAIL': GAIL,
        'BC': BehavioralCloning,
        'MyNewIRL': MyNewIRL,  # Add here
    }
    return algorithms[name]
```

5. **Add Tests**

```python
# tests/test_my_new_irl.py
def test_my_new_irl_training():
    """Test that MyNewIRL trains successfully"""
    from irl_training.my_new_irl import MyNewIRL

    # Setup
    demonstrations = load_test_demonstrations()
    env = gym.make('kaz-single-player-v0', type_of_player='knight')

    # Train
    irl = MyNewIRL(demonstrations, env)
    irl.train(n_epochs=5)

    # Validate
    policy = irl.get_policy()
    assert policy is not None

    print("✓ MyNewIRL test passed")
```

6. **Document Your Algorithm**

```markdown
# irl_training/README.md

### 7. **MyNewIRL** - `my_new_irl.py`

**Your Algorithm Name**

- **Type**: [IRL/Imitation Learning]
- **Key Idea**: Brief description
- **Advantages**: List advantages
- **Use Case**: When to use this algorithm

**References:**

- Your paper citation
```

## 🎮 Adding a New Environment

### Requirements

New environments must:

- Support OpenAI Gym interface
- Have two-player cooperative gameplay
- Provide distinct action spaces for each agent
- Return appropriate observations, rewards, done flags

### Implementation

1. **Create Environment File**

```python
# envs/my_new_env/my_environment.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyCooperativeEnv(gym.Env):
    """
    Your cooperative environment.

    Description of the game/task.
    """

    def __init__(self, owner_policy=None, **kwargs):
        super().__init__()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(5)  # Example
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        self.owner_policy = owner_policy

    def reset(self):
        """Reset environment to initial state"""
        # Reset logic
        return observation

    def step(self, action):
        """Execute one timestep"""
        # Step logic for assistant action
        # Execute owner policy action
        # Compute reward
        # Check termination
        return observation, reward, done, info

    def render(self, mode='human'):
        """Render the environment"""
        # Rendering logic
        pass
```

2. **Register Environment**

```python
# common/common.py
def register_environments():
    gym.register(
        id='my-cooperative-env-v0',
        entry_point='envs.my_new_env.my_environment:MyCooperativeEnv',
        max_episode_steps=1000,
    )
```

3. **Add Configuration**

```yaml
# configuration/main_experiment_config.yaml
environment:
  env_id: "my-cooperative-env-v0"
  env_kwargs:
    # Environment-specific parameters
```

4. **Create Tests**

```python
# tests/environment_tests/test_my_environment.py
def test_my_environment():
    """Test environment functionality"""
    from common.common import register_environments
    register_environments()

    env = gym.make('my-cooperative-env-v0')
    obs = env.reset()

    # Test observation space
    assert obs.shape == env.observation_space.shape

    # Test step
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    env.close()
    print("✓ Environment test passed")
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# All tests
cd tests
python3 test_classic_gym_game_set_up.py
python3 test_full_framework_functionality.py

# Environment tests
cd environment_tests
python3 test_fixed_horizon_kaz_works.py
python3 test_variable_horizon_kaz_works.py

# Manual gameplay
cd manual_gameplay
python3 play_different_envs.py
```

### Writing Tests

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Regression tests**: Ensure changes don't break existing functionality
- **Visual tests**: Manual gameplay verification

### Test Coverage Goals

- All new algorithms should have tests
- New environments should have comprehensive tests
- Critical paths should be covered
- Edge cases should be tested

## 📝 Code Style

### Python Style Guide

Follow PEP 8 with these specifics:

```python
# Imports
import standard_library
import third_party_library
from local_module import LocalClass

# Naming
class MyClass:  # PascalCase for classes
    def my_method(self):  # snake_case for functions/methods
        my_variable = 5  # snake_case for variables
        MY_CONSTANT = 10  # UPPER_CASE for constants

# Docstrings
def my_function(param1, param2):
    """
    Brief description.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass

# Type hints (encouraged)
def process_data(data: np.ndarray) -> dict:
    return {}
```

### Documentation Style

- Clear, concise descriptions
- Include examples where helpful
- Link to relevant papers/resources
- Explain "why" not just "what"

## 🔍 Code Review Process

### Before Submitting PR

1. **Run Tests**

```bash
cd tests
python3 test_full_framework_functionality.py
```

2. **Format Code**

```bash
# Optional but recommended
pip install black
black code/src/
```

3. **Update Documentation**

- Add/update docstrings
- Update relevant README files
- Add examples if applicable

4. **Check for Issues**

```bash
# Optional: Use linter
pip install flake8
flake8 code/src/
```

### Pull Request Guidelines

**PR Title Format:**

```
[Category] Brief description

Categories: Feature, Bugfix, Docs, Test, Refactor
Example: [Feature] Add MaxEnt IRL algorithm
```

**PR Description Template:**

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

Describe testing performed

## Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No breaking changes (or documented)

## Related Issues

Closes #issue_number
```

## 🐛 Bug Reports

### Good Bug Report

Include:

1. **Description**: Clear description of the bug
2. **Reproduction**: Steps to reproduce
3. **Expected**: What should happen
4. **Actual**: What actually happens
5. **Environment**: OS, Python version, GPU/CPU
6. **Logs**: Relevant error messages

**Example:**

```markdown
## Bug: AIRL training fails with CUDA out of memory

**Description:**
When training AIRL with n_envs=16, get CUDA OOM error.

**Steps to Reproduce:**

1. Set n_envs=16 in configuration
2. Run `python3 experiments/run_main.py`
3. Error occurs after ~100 epochs

**Expected:** Should train successfully
**Actual:** CUDA out of memory error

**Environment:**

- Ubuntu 20.04
- Python 3.8.10
- NVIDIA RTX 2080 (8GB)
- CUDA 11.3

**Error Log:**
```

RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB...

```

**Possible Fix:**
Reduce batch size or n_envs
```

## 💡 Feature Requests

### Good Feature Request

Include:

1. **Use Case**: Why is this needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered?
4. **Additional Context**: Any relevant info

## 🎓 Research Contributions

### Sharing Experimental Results

If you've run experiments:

1. **Document Hyperparameters**

```yaml
# Save your configuration
experiment:
  name: "My_Experiment"
  description: "Testing new IRL algorithm"
  # ... all parameters
```

2. **Share Results**

- Weights & Biases runs
- Performance graphs
- Statistical analysis
- Trained models (if small enough)

3. **Write Up**

- Brief description of experiment
- Key findings
- Comparison with baselines
- Lessons learned

## 📞 Getting Help

- **Questions**: Open a GitHub issue with "Question" label
- **Discussions**: Use GitHub Discussions
- **Bugs**: Open issue with detailed report
- **Ideas**: Open issue or discussion

## 🙏 Recognition

Contributors will be:

- Listed in README.md
- Credited in any publications using their contributions
- Acknowledged in commit messages

## 📄 License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🌟 Thank You!

Your contributions help advance research in cooperative AI and human-robot interaction. Every contribution, no matter how small, is valuable!

---

_Happy Contributing! 🚀_
