# IRL and Imitation Learning Implementations

This directory contains implementations of various Inverse Reinforcement Learning (IRL) and imitation learning algorithms used for human intention recognition. All implementations are adapted from the [Imitation](https://imitation.readthedocs.io/) library and follow a standardized interface for plug-and-play experimentation.

## 🎯 Purpose

These algorithms enable the assistant agent to learn from expert demonstrations (the owner's behavior) and infer the underlying intentions/reward function. This learned understanding helps the assistant anticipate and support the owner's actions.

---

## 📁 Available Algorithms

### 1. **AIRL** - `airl.py`

**Adversarial Inverse Reinforcement Learning**

- **Type**: Model-based IRL
- **Key Idea**: Uses adversarial training (like GANs) to learn a reward function that makes expert demonstrations optimal
- **Advantages**:
  - Recovers true reward function (under certain assumptions)
  - Robust to environment dynamics changes
  - Works well with function approximation
- **Use Case**: When you need accurate reward recovery for intention understanding

**Core Components:**

```python
from irl_training.airl import AIRL

airl = AIRL(
    demonstrations=expert_demos,
    env=training_env,
    gen_algo='ppo'  # Generator RL algorithm
)
airl.train(n_epochs=100)
reward_function = airl.get_reward()
learned_policy = airl.get_policy()
```

**How It Works:**

1. Discriminator learns to distinguish expert from generated trajectories
2. Generator (RL agent) tries to fool the discriminator
3. Converges to a reward function where expert behavior is optimal

**References:**

- Fu et al. (2017) - "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"

---

### 2. **GAIL** - `gail.py`

**Generative Adversarial Imitation Learning**

- **Type**: Direct policy learning (no explicit reward)
- **Key Idea**: Directly learns policy via adversarial training without recovering reward function
- **Advantages**:
  - Sample efficient
  - No need for reward engineering
  - Works well in high-dimensional spaces
- **Use Case**: When you care more about matching behavior than understanding rewards

**Core Components:**

```python
from irl_training.gail import GAIL

gail = GAIL(
    demonstrations=expert_demos,
    env=training_env,
    gen_algo='ppo'
)
gail.train(n_epochs=100)
imitation_policy = gail.get_policy()
```

**How It Works:**

1. Discriminator learns to classify expert vs. learner state-action pairs
2. Policy trained to maximize discriminator confusion
3. Directly imitates expert without explicit reward recovery

**References:**

- Ho & Ermon (2016) - "Generative Adversarial Imitation Learning"

---

### 3. **Behavioral Cloning (BC)** - `bcloning.py`

**Supervised Learning Approach**

- **Type**: Direct policy learning
- **Key Idea**: Supervised learning - train policy to predict expert actions given states
- **Advantages**:
  - Simple and fast
  - No environment interaction needed during training
  - Stable training process
- **Disadvantages**:
  - Suffers from distribution shift
  - No correction mechanism for errors
- **Use Case**: Baseline comparison, quick prototyping

**Core Components:**

```python
from irl_training.bcloning import BehavioralCloning

bc = BehavioralCloning(
    demonstrations=expert_demos,
    policy_network='MlpPolicy'
)
bc.train(n_epochs=50)
cloned_policy = bc.get_policy()
```

**How It Works:**

1. Treat expert demonstrations as supervised learning dataset
2. Train neural network: state → action mapping
3. Minimize prediction error (e.g., cross-entropy)

**References:**

- Pomerleau (1991) - "Efficient Training of Artificial Neural Networks for Autonomous Navigation"

---

### 4. **DAgger** - `DAgger.py`

**Dataset Aggregation**

- **Type**: Interactive imitation learning
- **Key Idea**: Iteratively collect more data from expert to handle distribution shift
- **Advantages**:
  - Addresses BC's distribution shift problem
  - Provably reduces compounding errors
  - More robust than pure BC
- **Disadvantages**:
  - Requires online expert feedback
  - More computationally expensive than BC
- **Use Case**: When expert is available for interactive queries

**Core Components:**

```python
from irl_training.DAgger import DAgger

dagger = DAgger(
    demonstrations=initial_demos,
    env=training_env,
    expert_policy=expert  # For interactive queries
)
dagger.train(n_rounds=10)
robust_policy = dagger.get_policy()
```

**How It Works:**

1. Train initial policy via BC
2. Execute policy and collect visited states
3. Query expert for actions on these states
4. Add to dataset and retrain
5. Repeat until convergence

**References:**

- Ross et al. (2011) - "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"

---

### 5. **Density-Based Reward Modeling** - `density_based_reward_modelling.py`

**Kernel Density Estimation Approach**

- **Type**: Reward learning via density estimation
- **Key Idea**: Assumes expert visits high-reward states more frequently
- **Advantages**:
  - Simple and interpretable
  - No adversarial training needed
  - Fast to train
- **Use Case**: Quick reward approximation, exploratory analysis

**Core Components:**

```python
from irl_training.density_based_reward_modelling import DensityReward

density_irl = DensityReward(
    demonstrations=expert_demos,
    kernel='gaussian',
    bandwidth=0.5
)
density_irl.train()
reward_estimate = density_irl.get_reward()
```

**How It Works:**

1. Build kernel density estimate over expert state visitation
2. High-density regions assigned high reward
3. Use learned reward for RL training

**References:**

- Brown et al. (2019) - "Extrapolating Beyond Suboptimal Demonstrations via Inverse Reinforcement Learning from Observations"

---

### 6. **Preference Comparisons** - `pref_comp.py`

**Learning from Preferences**

- **Type**: Reward learning from comparative feedback
- **Key Idea**: Learn reward from trajectory comparisons rather than demonstrations
- **Advantages**:
  - Doesn't require expert demonstrations
  - Natural for human feedback
  - Handles partial information
- **Use Case**: When you have preference labels but not optimal trajectories

**Core Components:**

```python
from irl_training.pref_comp import PreferenceComparisons

pref_comp = PreferenceComparisons(
    env=training_env,
    preference_dataset=trajectory_pairs_with_labels
)
pref_comp.train(n_iterations=100)
reward_from_prefs = pref_comp.get_reward()
```

**How It Works:**

1. Present trajectory pairs to expert/user
2. Expert labels which trajectory is better
3. Learn reward function that matches these preferences
4. Use learned reward for policy training

**References:**

- Christiano et al. (2017) - "Deep Reinforcement Learning from Human Preferences"

---

## 🏗️ Architecture: Template Interface

All algorithms inherit from `irl_template.py` which provides a standardized interface:

### `IRLTemplate` Abstract Class

**Required Methods:**

```python
class IRLTemplate(ABC):
    @abstractmethod
    def train(self, **kwargs):
        """Train the IRL/imitation learning algorithm"""
        pass

    @abstractmethod
    def get_reward(self):
        """Return learned reward function (if applicable)"""
        pass

    @abstractmethod
    def get_policy(self):
        """Return learned/imitated policy"""
        pass

    @abstractmethod
    def save(self, path):
        """Save trained model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load trained model"""
        pass
```

**Benefits:**

- Consistent API across all algorithms
- Easy to swap algorithms in experiments
- Simplified testing and evaluation
- Clear contract for new implementations

---

## 🔄 How IRL Fits Into the System

### Training Pipeline

```
┌─────────────────────────────────────────────────────┐
│  1. COLLECT EXPERT DEMONSTRATIONS                   │
│     Owner agent plays KAZ → record state-action     │
│     trajectories → create demonstration dataset     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  2. IRL TRAINING                                    │
│     Pass demonstrations to IRL algorithm →          │
│     Learn reward function and/or policy             │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  3. EXTRACT OUTPUTS                                 │
│     • Learned reward function (AIRL, Density)       │
│     • Learned policy (all algorithms)               │
│     • Internal model representation                 │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  4. COMBINE WITH RL                                 │
│     IRL output + Assistant's RL policy →            │
│     MLP combiner → Final assistant actions          │
└─────────────────────────────────────────────────────┘
```

### In `Assistant.py`

```python
# Step 1: Train IRL model on owner demonstrations
self.irl_model.train(demonstrations=owner_trajectories)

# Step 2: Extract learned policy/reward
internal_model = self.irl_model.get_policy()

# Step 3: Combine with RL policy
# MLP takes both IRL and RL outputs as input
combined_action = self.action_combiner(
    irl_output=internal_model.predict(obs),
    rl_output=self.rl_policy.predict(obs)
)
```

---

## 📊 Algorithm Comparison

| Algorithm       | Type      | Reward Recovery | Sample Efficiency | Complexity | Best For             |
| --------------- | --------- | --------------- | ----------------- | ---------- | -------------------- |
| **AIRL**        | IRL       | ✅ Yes          | Medium            | High       | Reward understanding |
| **GAIL**        | Imitation | ❌ No           | High              | Medium     | Direct imitation     |
| **BC**          | Imitation | ❌ No           | Very High         | Low        | Quick baseline       |
| **DAgger**      | Imitation | ❌ No           | Medium            | Medium     | Interactive learning |
| **Density**     | IRL       | ✅ Approximate  | High              | Low        | Fast prototyping     |
| **Pref. Comp.** | IRL       | ✅ Yes          | Low               | Medium     | Human feedback       |

---

## 🔧 Usage Examples

### Basic Training

```python
from irl_training.airl import AIRL
from stable_baselines3 import PPO

# Collect demonstrations
demonstrations = collect_expert_trajectories(owner_policy, env, n_episodes=100)

# Initialize and train IRL
irl = AIRL(
    demonstrations=demonstrations,
    env=training_env,
    gen_algo='ppo',
    gen_algo_kwargs={'learning_rate': 3e-4}
)

# Train for specified epochs
irl.train(n_epochs=200)

# Extract results
learned_reward = irl.get_reward()
learned_policy = irl.get_policy()

# Save for later use
irl.save('models/owner_model_airl.pkl')
```

### Comparing Multiple Algorithms

```python
from irl_training import AIRL, GAIL, BehavioralCloning

algorithms = {
    'AIRL': AIRL(demonstrations, env),
    'GAIL': GAIL(demonstrations, env),
    'BC': BehavioralCloning(demonstrations)
}

results = {}
for name, algo in algorithms.items():
    algo.train()
    policy = algo.get_policy()
    performance = evaluate_policy(policy, env)
    results[name] = performance

print(f"Best algorithm: {max(results, key=results.get)}")
```

---

## 🐛 Troubleshooting

### Common Issues

**AIRL/GAIL not converging:**

- Increase number of training epochs
- Adjust generator learning rate
- Ensure demonstrations are high-quality
- Check discriminator/generator balance

**BC overfitting:**

- Add regularization (dropout, weight decay)
- Increase demonstration dataset size
- Use data augmentation

**Out of memory:**

- Reduce batch size
- Use gradient accumulation
- Train on CPU if GPU limited

**Poor imitation performance:**

- Verify demonstration quality
- Check state/action space alignment
- Ensure sufficient demonstration diversity
- Try different algorithms

---

## 📈 Evaluation Metrics

### For IRL Algorithms (AIRL, Density, Pref. Comp.)

```python
# Reward recovery accuracy
true_reward = env.get_true_reward()
learned_reward = irl.get_reward()
correlation = compute_reward_correlation(true_reward, learned_reward)

# Policy performance
mean_reward = evaluate_policy(learned_policy, env, n_episodes=100)
```

### For Imitation Algorithms (GAIL, BC, DAgger)

```python
# Behavioral cloning metrics
action_accuracy = compute_action_accuracy(policy, test_demos)
state_distribution_distance = compute_state_dist_kl(policy, expert)

# Performance metrics
expert_performance = evaluate_policy(expert_policy, env)
learned_performance = evaluate_policy(learned_policy, env)
performance_gap = expert_performance - learned_performance
```

---

## 🔬 Research Insights

### Why Multiple Algorithms?

Different algorithms excel in different scenarios:

- **AIRL**: Best when you need interpretable reward functions for understanding intentions
- **GAIL**: Optimal for direct behavior matching without caring about rewards
- **BC**: Quick baseline, works well with lots of data
- **DAgger**: When you can interact with expert during training
- **Density**: Fast approximation for exploratory analysis
- **Pref. Comp.**: When demonstrations aren't available but preferences are

### Key Findings from Dissertation

1. **Density-based methods outperformed expectations**: PPO-Density emerged as the best configuration, challenging the hypothesis that AIRL would be optimal
2. **Fixed horizons critical**: Dramatic performance improvement with fixed-horizon (3000 timesteps) environments
3. **Variable horizons problematic**: Mixed results suggest imitation library implementations may not be optimized for variable-length episodes
4. **Simple can be better**: Density-based reward modeling proved more effective than complex adversarial approaches in this domain
5. **Quality over quantity**: Demonstration quality matters more than quantity for all algorithms
6. **Convergence patterns**: Fixed horizons provide more stable learning signals for intention recognition

### Algorithm Performance Ranking (Fixed Horizon)

1. **PPO-Density** ⭐ - Best overall performance
2. **PPO-AIRL** - Strong performer, second place
3. **PPO-GAIL** - Competitive results
4. **PPO-BC** - Surprisingly effective baseline
5. Other combinations - Variable performance

_Note: Rankings based on cooperative performance in KAZ fixed-horizon environment with 3000 timesteps_

---

## 📚 References

### Papers

- **AIRL**: Fu et al. (2017) - "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning"
- **GAIL**: Ho & Ermon (2016) - "Generative Adversarial Imitation Learning"
- **DAgger**: Ross et al. (2011) - "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
- **Preference Comparisons**: Christiano et al. (2017) - "Deep Reinforcement Learning from Human Preferences"

### Libraries

- **Imitation Library**: https://imitation.readthedocs.io/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/

---

## 🚀 Adding New Algorithms

To implement a new IRL/imitation algorithm:

1. **Create new file**: `my_new_algorithm.py`

2. **Inherit from template**:

```python
from irl_training.irl_template import IRLTemplate

class MyNewAlgorithm(IRLTemplate):
    def __init__(self, demonstrations, env, **kwargs):
        self.demonstrations = demonstrations
        self.env = env
        # Initialize your algorithm

    def train(self, **kwargs):
        # Implement training logic
        pass

    def get_reward(self):
        # Return learned reward (if applicable)
        pass

    def get_policy(self):
        # Return learned policy
        pass

    def save(self, path):
        # Save model
        pass

    def load(self, path):
        # Load model
        pass
```

3. **Update configuration**: Add to `configuration/main_experiment_config.yaml`

4. **Test**: Use `tests/` framework to validate

---

This modular design enables rapid experimentation with different IRL approaches while maintaining code quality and reproducibility.
