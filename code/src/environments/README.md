# Knights, Archers, Zombies (KAZ) Environment Documentation

This directory contains all game environment implementations used for testing IRL-based intention recognition algorithms. The environments are built on PettingZoo's KAZ game and adapted for single-agent and multi-agent reinforcement learning using OpenAI Gym interfaces.

## 🎮 Game Overview

**Knights, Archers, Zombies (KAZ)** is a two-player cooperative game where:

- **Knight (Owner/Human Agent)**: Melee fighter with sword, represents the human needing assistance
- **Archer (Assistant Agent)**: Ranged fighter with bow, represents the robotic assistant
- **Objective**: Cooperatively defend against waves of zombies

The game creates an ideal testbed for assistive AI because:

1. Two distinct action spaces (movement + sword vs. movement + arrows)
2. Power imbalance favoring the assistant (archer is more effective at killing zombies)
3. Requires anticipatory cooperation for optimal performance
4. Similar dynamics to assisted living scenarios

### Why KAZ Works for This Research

**Key Insight from Design**: The archer agent naturally learns to protect the knight to maximize its own reward. Since the game ends when either player dies or a zombie reaches the other side, the archer is incentivized to:

- Keep the knight alive (prevents early termination)
- Defend against zombies (primary objective)
- Anticipate where the knight needs protection

This creates **emergent cooperative behavior** without explicitly programming it, making it ideal for testing whether IRL-based intention recognition can further enhance this cooperation.

### Adaptation from PettingZoo

Original PettingZoo KAZ supports simultaneous two-player training, but this research required a different approach:

1. **Human Independence**: In assisted living, the elderly person acts independently based on lifetime experience
2. **Solution**: Train knight in isolation (single-player) via deep RL → saves policy
3. **Assistant Training**: Treat trained knight as stochastic process in environment → train archer via OpenAI Gym
4. **Result**: Archer learns to cooperate with pre-trained knight, mirroring real assistive scenario

This design allows using OpenAI Gym's single-agent infrastructure while maintaining the cooperative two-player dynamics.

---

## 📁 Directory Structure

```
environments/
├── kaz/                   # KAZ game environments
│   ├── core/              # Vanilla KAZ implementations
│   │   ├── single_player.py
│   │   ├── double_player.py
│   │   ├── fixed_horizon_single_player.py
│   ├── fixed_horizon_double_player.py
│   ├── img/               # Pygame rendering assets
│   └── src/               # Core game mechanics
│       ├── constants.py
│       ├── img.py
│       ├── players.py
│       ├── weapons.py
│       ├── zombie.py
│       └── skill_levels/  # Difficulty configurations
│           ├── vanilla.py
│           ├── medium.py
│           └── hard.py
├── kaz/variants/          # Modified KAZ environments
│   ├── eval/              # Evaluation-specific variants
│   ├── fixed_horizon/     # Fixed episode length variants
│   └── variable_horizon/  # Variable episode length variants
└── README.md              # This file
```

---

## 🔧 Core Environments (`kaz/core/`)

### Single-Player Environments

#### `single_player.py`

**Variable-horizon KAZ for training one agent in isolation.**

- **Use Case**: Train owner or assistant independently
- **Episode Termination**: Game ends when player dies or zombie reaches the other side
- **Key Parameter**: `type_of_player` - specifies which agent to train (knight/archer)

**Example:**

```python
import gymnasium as gym
env = gym.make('kaz-single-player-v0', type_of_player='knight')
```

#### `fixed_horizon_single_player.py`

**Fixed-horizon variant for controlled episode length.**

- **Use Case**: Ablation studies on time horizon effects
- **Episode Termination**: Fixed number of timesteps (e.g., 3000)
- **Key Parameter**: `horizon_limit` - maximum episode length

**Example:**

```python
env = gym.make('kaz-fixed-single-player-v0',
               type_of_player='archer',
               horizon_limit=3000)
```

---

### Double-Player Environments

#### `double_player.py`

**Variable-horizon KAZ for cooperative two-agent training.**

- **Use Case**: Train assistant while owner follows pre-trained policy
- **Episode Termination**: Game ends naturally based on game state
- **Key Requirement**: Owner policy must be provided as constructor argument

**Example:**

```python
from stable_baselines3 import PPO

owner_policy = PPO.load("path/to/owner_model.zip")
env = gym.make('kaz-double-player-v0',
               owner_policy=owner_policy)
```

**Training Flow:**

1. Owner agent trained in single-player mode → saves policy
2. Owner policy loaded and passed to double-player environment
3. Assistant agent trained while owner follows learned policy

#### `fixed_horizon_double_player.py`

**Fixed-horizon variant for two-agent cooperative training.**

- **Use Case**: Controlled experiments with fixed episode lengths
- **Episode Termination**: Fixed timestep limit
- **Key Parameters**: `owner_policy`, `horizon_limit`

---

## 🎯 Skill Levels (`kaz/core/src/skill_levels/`)

Control game difficulty by modifying archer arrow speed. Slower arrows make it harder for the archer to protect the knight, requiring better anticipation and cooperation.

### `vanilla.py`

**Default difficulty**

- Archer arrow speed: 45
- Easiest setting for cooperation

### `medium.py`

**Moderate difficulty**

- Archer arrow speed: 35
- Balanced challenge for testing

### `hard.py`

**High difficulty**

- Archer arrow speed: 25
- Requires precise intention recognition

**Usage:**

```python
from environments.kaz_core.src.skill_levels import medium

env = gym.make('kaz-single-player-v0',
               type_of_player='archer',
               skill_level=medium)
```

---

## 🔀 Environment Variants (`kaz/variants/`)

### Fixed Horizon Variants

#### `fixed_horizon_full_obs_double_player.py`

**Full observability + fixed horizon**

- **Observation Space**: Global (entire game state visible)
- **vs. Local**: Standard KAZ uses local observations (limited field of view)
- **Use Case**: Test whether full information improves IRL performance

**Key Difference:**

```python
# Local observation (default): 30x30 pixel window around agent
# Global observation: Entire game grid visible
```

#### `fixed_horizon_random_double_player.py`

**Random action baseline + fixed horizon**

- **Behavior**: Ignores trained policies, samples random actions
- **Use Case**: Baseline for measuring minimum performance
- **Parameters**:
  - `share_random_policy=False`: Each agent acts randomly independently
  - `share_random_policy=True`: Both agents take same random action

#### `fixed_horizon_single_policy_double_player.py`

**Single policy controls both agents + fixed horizon**

- **Behavior**: One RL policy outputs actions for both knight and archer
- **Use Case**: Measure performance when full cooperation is enforced
- **Action Mapping**: If policy outputs "fire arrow", knight swings sword instead

---

### Variable Horizon Variants

Same as fixed horizon variants but episodes terminate naturally:

#### `full_obs_double_player.py`

Global observations + variable episode length

#### `random_double_player.py`

Random actions + variable episode length

#### `single_policy_double_player.py`

Single shared policy + variable episode length

---

### Evaluation Variants (`eval/`)

Specialized environments for evaluation and testing:

#### `double_player_eval.py`

Standard evaluation environment with metrics logging

#### `fixed_horizon_double_player_eval.py`

Fixed horizon evaluation with performance tracking

#### `fixed_horizon_full_obs_double_player_eval.py`

Fixed horizon + full observability for evaluation

#### `full_obs_double_player_eval.py`

Variable horizon + full observability for evaluation

**Evaluation Features:**

- Additional performance metrics
- Episode statistics tracking
- Visualization support
- Deterministic seeding for reproducibility

---

## 🎨 Rendering (`img/` and `src/img.py`)

### Visual Assets

The `img/` directory contains Pygame sprites for:

- Knight character and animations
- Archer character and animations
- Zombies
- Weapons (sword, arrows)
- Background tiles

### Rendering System (`src/img.py`)

Handles game visualization using Pygame:

- Sprite management
- Animation frames
- Screen rendering
- FPS control

**Enable Rendering:**

```python
env = gym.make('kaz-double-player-v0',
               owner_policy=policy,
               render_mode='human')  # or 'rgb_array'
```

---

## 🎯 Core Game Mechanics (`src/`)

### `constants.py`

Game configuration constants:

- Grid dimensions
- Movement speeds
- Weapon properties
- Collision detection parameters
- Reward structure

### `players.py`

Player (Knight and Archer) classes:

- Movement logic
- Health management
- Weapon handling
- Collision detection

### `weapons.py`

Weapon mechanics:

- Sword swing (melee)
- Arrow projectiles (ranged)
- Damage calculation
- Hit detection

### `zombie.py`

Zombie enemy implementation:

- Spawning logic
- Movement patterns
- Wave system
- Health and damage

---

## 🔄 How Environments are Used in Experiments

### Training Pipeline

1. **Owner Training (Single-Player)**

   ```
   single_player.py → train knight with PPO → save owner_policy.zip
   ```

2. **Assistant RL Training (Single-Player)**

   ```
   single_player.py → train archer with PPO → save assistant_rl_policy.zip
   ```

3. **IRL Training (Double-Player)**

   ```
   double_player.py + owner_policy → observe owner actions →
   train IRL model → extract learned reward/policy
   ```

4. **Combined Assistant Training (Double-Player)**

   ```
   double_player.py + owner_policy →
   combine IRL output + RL policy via MLP →
   train final assistant
   ```

5. **Evaluation (Eval Variants)**
   ```
   *_eval.py + owner_policy + final_assistant_policy →
   measure cooperative performance
   ```

---

## 📊 Environment Comparison Table

| Environment                      | Players | Horizon  | Observations | Primary Use          |
| -------------------------------- | ------- | -------- | ------------ | -------------------- |
| `single_player.py`               | 1       | Variable | Local        | Individual training  |
| `double_player.py`               | 2       | Variable | Local        | Cooperative training |
| `fixed_horizon_single_player.py` | 1       | Fixed    | Local        | Ablation study       |
| `fixed_horizon_double_player.py` | 2       | Fixed    | Local        | Ablation study       |
| `full_obs_double_player.py`      | 2       | Variable | Global       | Information study    |
| `random_double_player.py`        | 2       | Variable | Local        | Baseline             |
| `single_policy_double_player.py` | 2       | Variable | Local        | Upper bound          |

---

## 🛠️ Creating Custom Environments

To add a new environment variant:

1. **Inherit from base class:**

   ```python
   from environments.kaz_core.double_player import DoublePlayerKAZ

   class CustomKAZ(DoublePlayerKAZ):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
   ```

2. **Override methods as needed:**

   ```python
   def step(self, action):
       # Custom step logic
       obs, reward, done, info = super().step(action)
       # Modify observation, reward, etc.
       return obs, reward, done, info

   def reset(self):
       # Custom reset logic
       obs = super().reset()
       return obs
   ```

3. **Register environment:**
   ```python
   # In common/common.py
   gym.register(
       id='custom-kaz-v0',
       entry_point='envs.custom_kaz:CustomKAZ'
   )
   ```

---

## 🐛 Common Issues

**Issue: Environment doesn't render**

- Check `render_mode` parameter
- Ensure Pygame is installed: `pip install pygame`

**Issue: Observation space mismatch**

- Verify environment variant matches expected obs space
- Check if using local vs. global observations

**Issue: Episode never terminates**

- For fixed horizon: ensure `horizon_limit` is set
- For variable horizon: check game-ending conditions

**Issue: Owner policy errors in double-player mode**

- Ensure owner policy observation space matches environment
- Verify policy was trained on compatible environment

---

## 📚 References

- **Original PettingZoo KAZ**: https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/
- **OpenAI Gym**: https://www.gymlibrary.dev/
- **Stable-Baselines3 Env Checker**: Use `check_env()` to validate custom environments

---

## 🔬 Research Insights

### Why These Variants Matter

1. **Fixed vs. Variable Horizon**: Tests whether episode length affects intention recognition
2. **Full vs. Local Observations**: Evaluates information requirements for cooperation
3. **Random Actions**: Establishes minimum performance baseline
4. **Single Policy**: Provides theoretical maximum cooperation upper bound
5. **Skill Levels**: Tests robustness across difficulty levels

These variants enable comprehensive ablation studies and ensure findings are robust across different environmental conditions.

## eval

This directory stores KAZ environments **_only_** used for evaluating the performance of learned policies. Rather than returning the reward of the `archer` like the other double-player KAZ implementations, these methods return the **_combined_** reward of the two agents per timestep. Therefore, this gives a better overall measure of their combined game performance rather than measuring it by the `archer's reward`.

- `double_player_eval.py` = Variable horizon, double player KAZ environment where returned reward formula has been updated as described above.

- `fixed_horizon_double_player_eval.py` = Fixed-horizon equivalent of `double_player_eval.py`.

- `full_obs_double_player_eval.py` = Variable horizon, double player, **_combined reward_** KAZ environment, which gives the `archer` agent a full game observation per timestep rather than a local observation.

- `fixed_horizon_full_obs_double_player_eval.py` = Fixed-horizon equivalent of `fixed_horizon_double_player_eval.py`.

## Links

- PettingZoo (https://pettingzoo.farama.org/environments/butterfly/knights_archers_zombies/)

- Pygame (https://www.pygame.org/news)
