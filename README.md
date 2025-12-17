# Investigating IRL-based Intention Recognition for Assistive Action Planning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Master's Dissertation Project** - Evaluating Inverse Reinforcement Learning algorithms for human intention recognition in cooperative multi-agent systems.

## 📋 Overview

This research project investigates whether **Inverse Reinforcement Learning (IRL)** based intention recognition algorithms can improve human-robot cooperation in assistive systems. Inspired by how 18-month-old children can infer human intentions and offer anticipatory help, this work explores embedding similar capabilities into AI assistants.

### Key Contributions

- 🎯 **Novel IRL-based Assistive Agent**: An agent that learns to anticipate and assist a human partner in cooperative tasks
- 🎮 **Flexible Testing Framework**: Environment-independent infrastructure for evaluating cooperative IRL algorithms
- 📊 **Comprehensive Evaluation**: Baseline experiments and ablation studies across multiple game difficulty levels and time horizons
- 🔧 **Modular Architecture**: Plug-and-play system supporting different RL/IRL algorithms and game environments

### Research Context

**Application Domain**: Assisted living for elderly populations  
**Test Environment**: Knights, Archers, Zombies (KAZ) - a two-player cooperative game  
**Core Question**: Can IRL-based intention recognition improve anticipatory assistance compared to traditional approaches?

## 🏗️ Architecture

The system consists of three main components:

```
┌─────────────────────────────────────────────────────────┐
│              IRL-BASED ASSISTANT AGENT                  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐            │
│  │  IRL Intention  │  │   Deep RL        │            │
│  │  Recognition    │  │   Environment    │            │
│  │  Module         │  │   Learning       │            │
│  └────────┬────────┘  └────────┬─────────┘            │
│           │                     │                       │
│           └──────────┬──────────┘                       │
│                      ▼                                  │
│            ┌─────────────────────┐                     │
│            │   Deep RL Action    │                     │
│            │   Prediction Module │                     │
│            │   (MLP Combiner)    │                     │
│            └─────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

**Owner/Human Agent** ← Trained via Deep RL  
**Assistant Agent** ← Trained via IRL + Deep RL + Action Combiner

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/JoshWheeler08/Investigating-IRL-based-intention-recognition-algorithms.git
cd Investigating-IRL-based-intention-recognition-algorithms
```

2. **Create a virtual environment**

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**

```bash
pip3 install -r requirements.txt
```

4. **Configure environment variables (Optional for WandB tracking)**

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your WandB API key
# Get your key from: https://wandb.ai/authorize
```

See [SECURITY.md](SECURITY.md) for secure credential management.

### Running Experiments

#### Baseline Experiments

Test the KAZ environment with varying levels of agent cooperation:

```bash
cd code/src
python3 experiments/run_baselines.py configuration/kaz_baselines_config.yaml
# Or use the convenience script:
./run_kaz_baseline_tests.sh
```

#### Main IRL Experiments

Evaluate IRL-based assistive agents:

```bash
cd code/src
python3 run_main_experiment.py configuration/main_experiment_config.yaml
# Or use the convenience script:
./run_main_experiment_tests.sh
```

### Configuration

Experiment hyperparameters can be modified in YAML configuration files:

- **Baseline experiments**: `code/src/configuration/kaz_baselines_config.yaml`
- **Main IRL experiments**: `code/src/configuration/main_experiment_config.yaml`

## 📁 Project Structure

```
├── code/
│   ├── requirements.txt          # Python dependencies
│   └── src/
│       ├── main_experiment_core.py      # Core experiment logic
│       ├── run_main_experiment.py       # IRL experiment runner
│       ├── run_kaz_baselines.py         # Baseline experiment runner
│       ├── baselines/                   # Baseline algorithms
│       │   └── kaz/                     # KAZ-specific baselines
│       ├── configuration/               # Experiment configs
│       ├── envs/                        # Game environments
│       │   ├── kaz_core/               # Core KAZ implementations
│       │   └── kaz_variants/           # Modified KAZ variants
│       ├── irl_training/                # IRL algorithm implementations
│       │   ├── airl.py                 # Adversarial IRL
│       │   ├── gail.py                 # Generative Adversarial Imitation
│       │   ├── bcloning.py             # Behavioral Cloning
│       │   ├── DAgger.py               # Dataset Aggregation
│       │   └── ...
│       ├── MyAgents/                    # Agent abstractions
│       │   ├── Owner.py                # Human/Owner agent
│       │   ├── Assistant.py            # IRL-based assistant
│       │   └── Objects/                # Environment/Policy objects
│       ├── Wrappers/                    # OpenAI Gym wrappers
│       └── common/                      # Shared utilities
├── tests/                               # Test suites and examples
│   ├── manual_gameplay/                # Manual gameplay testing
│   └── environment_tests/              # Environment unit tests
├── docker/                              # Docker configuration
└── README.md                            # This file
```

## 🔬 Experiments

### Baseline Tests (6 variants)

Tests varying levels of cooperation to establish performance bounds:

1. **Random actions** - Both agents act randomly
2. **Random shared policy** - Both agents follow the same random policy
3. **Single trained policy** - One RL policy controls both agents
4. **Knight only trained** - Only the human/knight is trained
5. **Archer only trained** - Only the assistant/archer is trained
6. **Separate policies** - Both agents trained independently

### Main IRL Experiments

- **IRL algorithms tested**: AIRL, GAIL, Behavioral Cloning, DAgger, Preference Comparisons, Density-based Reward Modeling
- **RL algorithms used**: PPO, A2C, DQN
- **Ablation studies**:
  - Variable vs. fixed time horizons
  - Multiple game difficulty levels (arrow speeds)
  - Full vs. partial observability

## 🛠️ Technologies Used

### Core Libraries

**[OpenAI Gym](https://www.gymlibrary.dev/)** (2016)

- Standard interface for single-agent RL research
- Provides benchmark environments across multiple domains (Atari, MuJoCo, Board Games)
- Custom environment support via standardized API
- Modular `Wrapper` classes for observation/action/reward preprocessing
- Vectorized environments for parallel training acceleration
- Enables training of human and assistant agents independently

**[PettingZoo](https://pettingzoo.farama.org/)** (2021)

- Multi-agent RL toolkit with Gym-compatible API
- Provides Knights, Archers, Zombies (KAZ) cooperative environment
- Modified to work with single-agent training via OpenAI Gym interface
- Built on Pygame for game rendering

**[Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/)**

- Reliable PyTorch implementations of deep RL algorithms
- Unified structure for easy algorithm switching
- Provides PPO, A2C, DQN (all used in this research)
- Automatic policy evaluation for performance metrics
- Supports discrete and continuous action spaces

**[Imitation](https://imitation.readthedocs.io/)**

- Comprehensive IRL and imitation learning library
- Implementations: AIRL, GAIL, DAgger, BC, Preference Comparisons
- Integrates seamlessly with Stable-Baselines3
- Used for intention recognition module

**[Weights & Biases](https://wandb.ai/)**

- MLOps platform for experiment tracking
- Real-time training visualization and monitoring
- Model versioning and dataset management
- Performance metric storage and comparison
- Hyperparameter optimization support
- Used extensively for organizing and evaluating all experiments

**[Pygame](https://www.pygame.org/)**

- Python modules for video game development
- Provides KAZ environment rendering
- Original PettingZoo implementation (unmodified)

### Development Tools

- **Version Control**: Git, GitHub
- **Package Management**: pip, venv
- **Containerization**: Docker + NVIDIA NGC Deep Learning Containers
- **Deep Learning**: PyTorch (backend for SB3)
- **Numerical Computing**: NumPy, SciPy

### Why These Technologies?

1. **OpenAI Gym**: Industry standard for RL, enables single-agent training of multi-agent environment
2. **Stable-Baselines3**: Reliable, optimized implementations save development time
3. **Imitation**: Only comprehensive library for multiple IRL algorithms
4. **Weights & Biases**: Essential for tracking long-running experiments and identifying convergence
5. **PettingZoo**: Provides cooperative game environment with appropriate dynamics

## 📊 Key Results

The research revealed important insights about IRL-based intention recognition:

### Main Findings

**Variable Horizon Environments:**

- IRL-enabled assistive agents showed mixed results, unable to consistently outperform all baseline experiments
- Suggests the need for optimization of imitation library implementations for variable-length episodes

**Fixed Horizon Environments:**

- **Dramatic performance improvement** when using fixed-horizon (3000 timesteps) KAZ environment
- IRL-based assistants demonstrated significantly better cooperation with the human agent
- Fixed horizons may provide more stable learning signals for intention recognition

**Best Configuration:**

- **PPO-Density emerged as the top performer** among tested RL-IRL combinations
- Unexpectedly outperformed PPO-AIRL, challenging initial hypotheses
- Density-based reward modeling proved more effective than anticipated

### Key Insights

- **Time horizon matters**: Fixed-length episodes significantly improved IRL algorithm performance
- **Simplicity wins**: Density-based methods proved competitive with more complex adversarial approaches
- **Environment design**: The archer naturally learns to protect the knight to maximize its own reward
- **Cooperation levels**: Varying cooperation in baselines successfully established performance bounds

_For detailed statistical analysis and experimental methodology, please refer to the full dissertation document._

## 🎯 Research Objectives

### ✅ Completed Primary Objectives

- [x] Modified KAZ environment for single/double-player RL training
- [x] Created comprehensive baseline experiment suite
- [x] Designed and implemented IRL-based assistant agent architecture
- [x] Evaluated performance across multiple metrics
- [x] Compared IRL approach against baselines with statistical significance

### ✅ Completed Secondary Objectives

- [x] Ablation study: Game difficulty levels
- [x] Ablation study: Fixed vs. variable time horizons
- [x] Developed flexible, plug-and-play testing infrastructure
- [x] Multiple IRL algorithms (AIRL, GAIL, etc.)

### 🔄 Future Work

#### Short Term

- [ ] **Generalization Testing**: Evaluate assistants on additional two-player cooperative games (e.g., 2-player Mario Bros)
- [ ] **Hyperparameter Optimization**: Explore additional RL-IRL configurations to potentially exceed PPO-Density performance
- [ ] **Algorithm Variants**: Test additional IRL algorithms and RL policy architectures
- [ ] **Statistical Robustness**: Increase number of random seeds for more robust statistical analysis

#### Medium Term

- [ ] **Pixel-Based Learning**: Redesign assistants to learn from pixel observations rather than state vectors
- [ ] **Vision Integration**: Add CNN-based perception for camera input processing
- [ ] **Real-World Sim-to-Real**: Bridge simulation results to real robotic platforms
- [ ] **Multi-Agent Deep RL**: Compare with MADRL approaches (baseline upper bound)
- [ ] **Publication**: Distill findings into peer-reviewed conference/journal paper

#### Long Term

- [ ] **Real-World Deployment**: Model assisted living context with human participants and autonomous robots
- [ ] **Video-Based Intention Recognition**: Apply algorithms with real video input of human actions
- [ ] **Anticipatory Assistance Validation**: Measure if robots provide desired assistance in practice
- [ ] **Clinical Studies**: Partner with assisted living facilities for real-world validation
- [ ] **Generalized Framework**: Extend to other human-robot cooperation domains

## 📚 Documentation

- **[Code Documentation](code/src/README.md)** - Detailed module descriptions
- **[Environment Guide](code/src/envs/README.md)** - KAZ environment variants
- **[IRL Algorithms](code/src/irl_training/README.md)** - IRL implementation details
- **[Agent Architecture](code/src/MyAgents/README.md)** - Owner and Assistant agents
- **[Testing Guide](tests/README.md)** - How to test and validate changes

## 🐳 Docker Support

For reproducible experiments on GPU systems:

```bash
cd docker
docker build -t irl-intention-recognition .
docker run --gpus all -v $(pwd):/workspace irl-intention-recognition
```

See `docker/start_container_commands/` for pre-configured experiment scripts.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{wheeler2023irl,
  title={Investigating Human Intention Recognition Algorithms based on Inverse Reinforcement Learning for Assistive Action Planning},
  author={Wheeler, Joshua},
  year={2023},
  school={University of St Andrews}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PettingZoo** team for the Knights, Archers, Zombies environment
- **Stable-Baselines3** and **Imitation** library developers
- **Warneken and Tomasello** for foundational research on human altruistic behavior

## 👤 Author

**Joshua Wheeler**  
Master's Dissertation Project  
University of St Andrews  
[GitHub](https://github.com/JoshWheeler08)

---

_This project was developed as part of a Master's dissertation investigating cooperative AI and human-robot interaction in assistive contexts._
