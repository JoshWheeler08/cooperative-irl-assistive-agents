# Agent Architecture Documentation

This directory contains the core agent abstractions that enable flexible, environment-independent testing of IRL-based assistive systems. The architecture separates concerns between the Owner/Human agent and the Assistant agent while providing a unified interface through inheritance.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Agent (ABC)                       │
│  • Common interface and shared methods              │
│  • train_rl(), predict(), save(), load()           │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼──────────┐  ┌──────▼────────────────────────┐
│  Owner.py        │  │  Assistant.py                 │
│  (Human/Owner)   │  │  (IRL-based Assistant)        │
│                  │  │                               │
│  • RL training   │  │  • RL training                │
│  • Simple policy │  │  • IRL training               │
│                  │  │  • Policy combination (MLP)   │
└──────────────────┘  └───────────────────────────────┘
```

---

## 📁 Core Files

### `Agent.py` - Abstract Base Class

**Purpose**: Defines the standard interface that all agents must implement.

**Key Methods:**

```python
class Agent(ABC):
    @abstractmethod
    def train_rl(self, env_obj, policy, policy_type):
        """Train agent using reinforcement learning"""
        pass

    @abstractmethod
    def predict(self, observation):
        """Predict action given observation"""
        pass

    @abstractmethod
    def save(self, path):
        """Save trained model"""
        pass

    @abstractmethod
    def load(self, path):
        """Load trained model"""
        pass

    # Shared utility methods
    def set_seed(self, seed):
        """Set random seed for reproducibility"""
        pass

    def get_performance_metrics(self):
        """Return training/evaluation metrics"""
        pass
```

**Benefits:**

- Enforces consistent interface across agent types
- Enables polymorphic usage in experiments
- Simplifies testing and debugging
- Facilitates extension to new agent types

---

### `Owner.py` - Human/Owner Agent

**Purpose**: Represents the human player (knight in KAZ) who needs assistance.

**Characteristics:**

- Trained using standard deep RL (PPO, A2C, DQN)
- No access to assistant's state or actions during training
- Learns to play optimally in single-player mode
- Policy is frozen when used in cooperative experiments

**Key Implementation Details:**

```python
class Owner(Agent):
    def __init__(self):
        self.rl_policy = None
        self.training_metrics = {}

    def train_rl(self, env_obj, policy='PPO', policy_type='MlpPolicy'):
        """
        Train owner in single-player environment

        Args:
            env_obj: EnvObject containing environment configuration
            policy: RL algorithm name ('PPO', 'A2C', 'DQN')
            policy_type: Policy network architecture
        """
        # Create RL model
        model = get_rl_algorithm(policy)(
            policy_type,
            env_obj.env,
            learning_rate=3e-4,
            verbose=1
        )

        # Train for specified timesteps
        model.learn(total_timesteps=env_obj.total_timesteps)

        self.rl_policy = model
        return model

    def predict(self, observation, deterministic=True):
        """
        Predict action for given observation

        Returns:
            action: Action to take
            _states: Internal policy states (if any)
        """
        return self.rl_policy.predict(observation, deterministic=deterministic)
```

**Training Pipeline:**

1. Create single-player environment (knight only)
2. Train using deep RL until convergence
3. Save trained policy
4. Use as fixed policy in cooperative experiments

**Example Usage:**

```python
from agents.owner import Owner
from agents.helpers.env_config import EnvObject

# Create owner agent
owner = Owner()

# Configure environment
env_obj = EnvObject(
    env_id='kaz-single-player-v0',
    n_envs=4,
    total_timesteps=1_000_000,
    env_kwargs={'type_of_player': 'knight'}
)

# Train owner
owner.train_rl(env_obj, policy='PPO', policy_type='MlpPolicy')

# Save for later use
owner.save('models/owner_policy.zip')
```

---

### `Assistant.py` - IRL-based Assistant Agent

**Purpose**: Represents the robotic assistant (archer in KAZ) that learns to help the owner.

**Characteristics:**

- Trained using **three components**:
  1. Independent RL policy (learns environment dynamics)
  2. IRL model (learns owner's intentions)
  3. MLP combiner (integrates both for final actions)
- Observes owner's behavior to build internal model
- Makes anticipatory decisions to assist owner
- More complex than Owner due to multi-stage training

**Architecture:**

```
                    ┌─────────────────┐
                    │   Observation   │
                    └────────┬────────┘
                             │
                ┌────────────┴───────────┐
                │                        │
        ┌───────▼────────┐     ┌────────▼────────┐
        │  RL Policy     │     │  IRL Model      │
        │  (Environment  │     │  (Owner's       │
        │   Dynamics)    │     │   Intentions)   │
        └───────┬────────┘     └────────┬────────┘
                │                        │
                └────────────┬───────────┘
                             │
                    ┌────────▼────────┐
                    │  MLP Combiner   │
                    │  (Action Fusion)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Action   │
                    └─────────────────┘
```

**Key Implementation Details:**

```python
class Assistant(Agent):
    def __init__(self):
        self.rl_policy = None           # Independent RL policy
        self.irl_model = None           # Intention recognition model
        self.action_combiner = None     # MLP for combining outputs
        self.training_metrics = {}

    def train_rl(self, env_obj, policy='PPO', policy_type='MlpPolicy'):
        """
        Stage 1: Train assistant's RL policy independently
        Learns how to control archer in single-player mode
        """
        model = get_rl_algorithm(policy)(
            policy_type,
            env_obj.env,
            learning_rate=3e-4
        )
        model.learn(total_timesteps=env_obj.total_timesteps)
        self.rl_policy = model
        return model

    def train_irl(self, demonstrations, irl_algorithm='AIRL', **kwargs):
        """
        Stage 2: Train IRL model on owner demonstrations
        Learns to recognize owner's intentions

        Args:
            demonstrations: Owner's state-action trajectories
            irl_algorithm: Algorithm choice ('AIRL', 'GAIL', 'BC', etc.)
        """
        from irl_training import get_irl_algorithm

        irl_model = get_irl_algorithm(irl_algorithm)(
            demonstrations=demonstrations,
            **kwargs
        )
        irl_model.train()
        self.irl_model = irl_model
        return irl_model

    def train_combiner(self, env_obj, owner_policy):
        """
        Stage 3: Train MLP combiner in cooperative environment
        Learns to weight RL and IRL outputs for optimal assistance

        Args:
            env_obj: Double-player environment configuration
            owner_policy: Trained owner policy
        """
        # Custom policy that combines RL and IRL
        class CombinerPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.rl_policy = self.rl_policy
                self.irl_model = self.irl_model

            def forward(self, obs):
                # Get predictions from both modules
                rl_action_probs = self.rl_policy.predict_proba(obs)
                irl_action_probs = self.irl_model.predict_proba(obs)

                # Combine via MLP
                combined = self.mlp(
                    torch.cat([rl_action_probs, irl_action_probs], dim=1)
                )
                return combined

        # Train combiner in cooperative environment
        combiner_model = PPO(
            CombinerPolicy,
            env_obj.env,
            learning_rate=3e-4
        )
        combiner_model.learn(total_timesteps=env_obj.total_timesteps)
        self.action_combiner = combiner_model
        return combiner_model

    def predict(self, observation, deterministic=True):
        """
        Predict action using full pipeline:
        RL output + IRL output → MLP → Final action
        """
        if self.action_combiner is not None:
            return self.action_combiner.predict(observation, deterministic)
        elif self.rl_policy is not None:
            return self.rl_policy.predict(observation, deterministic)
        else:
            raise ValueError("No policy trained yet!")
```

**Training Pipeline:**

1. **Stage 1 - Environment Learning (RL)**:

   - Train archer in single-player mode
   - Learn action space and environment dynamics
   - No owner interaction

2. **Stage 2 - Intention Recognition (IRL)**:

   - Observe owner playing in environment
   - Train IRL model on owner's trajectories
   - Extract learned policy/reward representing intentions

3. **Stage 3 - Cooperative Integration (MLP Combiner)**:
   - Train in double-player environment with owner
   - MLP learns to weight RL and IRL outputs
   - Optimizes for cooperative performance

**Example Usage:**

```python
from agents.assistant import Assistant
from agents.helpers.env_config import EnvObject

# Create assistant agent
assistant = Assistant()

# Stage 1: Train independent RL
rl_env_obj = EnvObject(
    env_id='kaz-single-player-v0',
    n_envs=4,
    total_timesteps=1_000_000,
    env_kwargs={'type_of_player': 'archer'}
)
assistant.train_rl(rl_env_obj, policy='PPO')

# Stage 2: Train IRL on owner demonstrations
owner_demos = collect_demonstrations(owner_policy, env, n_episodes=100)
assistant.train_irl(
    demonstrations=owner_demos,
    irl_algorithm='AIRL'
)

# Stage 3: Train combiner in cooperative setting
coop_env_obj = EnvObject(
    env_id='kaz-double-player-v0',
    n_envs=4,
    total_timesteps=500_000,
    env_kwargs={'owner_policy': owner_policy}
)
assistant.train_combiner(coop_env_obj, owner_policy)

# Save complete assistant
assistant.save('models/assistant_complete.zip')
```

---

## 📦 Objects Subdirectory

### `EnvObject.py` - Environment Encapsulation

**Purpose**: Bundles all environment-related configuration into a single object.

**Key Attributes:**

```python
class EnvObject:
    def __init__(
        self,
        env_id: str,              # Gym environment ID
        n_envs: int,              # Number of parallel environments
        total_timesteps: int,     # Total training timesteps
        env_kwargs: dict = {},    # Environment-specific arguments
        logger = None,            # Training logger
        seed: int = None          # Random seed
    ):
        self.env_id = env_id
        self.n_envs = n_envs
        self.total_timesteps = total_timesteps
        self.env_kwargs = env_kwargs
        self.logger = logger
        self.seed = seed

        # Create vectorized environment
        self.env = make_vec_env(
            env_id,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            seed=seed
        )
```

**Benefits:**

- Reduces function parameter clutter
- Ensures consistent environment configuration
- Simplifies environment recreation
- Enables easy serialization

**Example:**

```python
from agents.helpers.env_config import EnvObject

env_obj = EnvObject(
    env_id='kaz-double-player-v0',
    n_envs=8,
    total_timesteps=2_000_000,
    env_kwargs={
        'owner_policy': owner_policy,
        'skill_level': 'medium',
        'horizon_limit': 3000
    },
    seed=42
)

# Pass to agent training
assistant.train_combiner(env_obj, owner_policy)
```

---

### `PolicyObject.py` - Policy Encapsulation

**Purpose**: Bundles policy-related information and utilities.

**Key Attributes:**

```python
class PolicyObject:
    def __init__(
        self,
        policy_model,          # Trained policy (SB3 model)
        policy_type: str,      # Type ('PPO', 'A2C', 'DQN')
        training_env_id: str,  # Environment used for training
        metadata: dict = {}    # Additional information
    ):
        self.policy_model = policy_model
        self.policy_type = policy_type
        self.training_env_id = training_env_id
        self.metadata = metadata

    def predict(self, obs, deterministic=True):
        """Convenience method for predictions"""
        return self.policy_model.predict(obs, deterministic)

    def save(self, path):
        """Save policy and metadata"""
        self.policy_model.save(path)
        # Save metadata separately
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path, policy_type):
        """Load policy and metadata"""
        model = get_rl_algorithm(policy_type).load(path)
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        return cls(model, policy_type, metadata['env_id'], metadata)
```

**Benefits:**

- Tracks policy provenance (which env, which algorithm)
- Simplifies policy management
- Enables metadata attachment (hyperparameters, performance metrics)
- Consistent save/load interface

---

## 🔄 How Agents Work Together

### Experimental Flow

```
1. TRAIN OWNER (Owner.py)
   └─> Single-player environment
   └─> Deep RL (PPO/A2C/DQN)
   └─> Save owner_policy.zip

2. TRAIN ASSISTANT RL (Assistant.py - Stage 1)
   └─> Single-player environment
   └─> Deep RL (PPO/A2C/DQN)
   └─> Save assistant_rl_policy.zip

3. COLLECT OWNER DEMONSTRATIONS
   └─> Run owner_policy in environment
   └─> Record state-action trajectories
   └─> Save demonstration dataset

4. TRAIN ASSISTANT IRL (Assistant.py - Stage 2)
   └─> Pass owner demonstrations
   └─> IRL algorithm (AIRL/GAIL/etc.)
   └─> Extract internal model

5. TRAIN ASSISTANT COMBINER (Assistant.py - Stage 3)
   └─> Double-player environment
   └─> Owner policy + Assistant (RL + IRL)
   └─> Train MLP combiner
   └─> Save complete assistant

6. EVALUATE COOPERATION
   └─> Run owner + assistant in environment
   └─> Measure game performance
   └─> Compare with baselines
```

---

## 🎯 Design Principles

### 1. **Separation of Concerns**

- Owner handles human behavior
- Assistant handles assistance logic
- Agent base class handles shared functionality

### 2. **Plug-and-Play Architecture**

- Swap RL algorithms without changing agent code
- Swap IRL algorithms without changing experiment code
- Swap game environments without changing agent logic

### 3. **Encapsulation**

- EnvObject bundles environment config
- PolicyObject bundles policy info
- Reduces coupling, increases cohesion

### 4. **Extensibility**

- Easy to add new agent types
- Easy to add new training stages
- Easy to add new evaluation metrics

---

## 🚀 Advanced Usage

### Custom Agent Types

```python
from agents.agent_base import Agent

class MultiModalAssistant(Agent):
    """Assistant that uses vision + IRL"""

    def __init__(self):
        super().__init__()
        self.vision_model = None
        self.irl_model = None

    def train_vision(self, image_data):
        # Train vision component
        pass

    def train_rl(self, env_obj, policy, policy_type):
        # Custom RL training with vision
        pass
```

### Transfer Learning

```python
# Train owner on easy difficulty
owner.train_rl(easy_env_obj, policy='PPO')

# Fine-tune on hard difficulty
owner.rl_policy.learn(
    total_timesteps=100_000,
    env=hard_env
)
```

---

## 📚 References

- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Cooperative IRL (CIRL)**: Hadfield-Menell et al. (2016)
- **Object-Oriented RL**: Design patterns for RL systems

---

This architecture enables the systematic investigation of IRL-based intention recognition while maintaining code clarity, reusability, and extensibility.
