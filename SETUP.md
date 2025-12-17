# Setup and Installation Guide

This guide provides detailed instructions for setting up the IRL-based Intention Recognition framework on your system.

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [GPU Setup](#gpu-setup)
5. [Docker Setup](#docker-setup)
6. [Verifying Installation](#verifying-installation)
7. [Troubleshooting](#troubleshooting)
8. [Configuration](#configuration)

---

## 🖥️ System Requirements

### Minimum Requirements

- **OS**: Ubuntu 18.04+, macOS 10.14+, or Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum
- **Storage**: 5GB free space
- **CPU**: Modern multi-core processor

### Recommended for Training

- **OS**: Ubuntu 20.04 LTS
- **Python**: 3.8 - 3.10
- **RAM**: 16GB or more
- **Storage**: 20GB free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 2080, RTX 3080, A100, etc.)
- **CUDA**: 11.3 or higher

---

## 🚀 Quick Start

For experienced users who just want to get started:

```bash
# Clone repository
git clone https://github.com/JoshWheeler08/cooperative-irl-assistive-agents.git
cd cooperative-irl-assistive-agents

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
cd code/src
python3 -c "from common.common import register_environments; register_environments(); print('✓ Setup successful!')"

# Run baseline experiments
python3 run_kaz_baselines.py configuration/kaz_baselines_config.yaml
```

---

## 📦 Detailed Installation

### Step 1: Install Python

#### Ubuntu/Debian

```bash
sudo apt update
sudo apt install python3.8 python3.8-venv python3-pip
```

#### macOS

```bash
# Using Homebrew
brew install python@3.8
```

#### Windows

Download and install Python 3.8+ from [python.org](https://www.python.org/downloads/)

**Verify Python installation:**

```bash
python3 --version  # Should show Python 3.8.x or higher
```

### Step 2: Clone Repository

```bash
# Using HTTPS
git clone https://github.com/JoshWheeler08/cooperative-irl-assistive-agents.git

# Or using SSH
git clone git@github.com:JoshWheeler08/cooperative-irl-assistive-agents.git

# Navigate to directory
cd cooperative-irl-assistive-agents
```

### Step 3: Create Virtual Environment

**Why virtual environment?**

- Isolates project dependencies
- Prevents version conflicts
- Makes project portable

```bash
# Create virtual environment
python3 -m venv env

# Activate virtual environment
# On Linux/macOS:
source env/bin/activate

# On Windows:
env\Scripts\activate

# Your prompt should now show (env)
```

**To deactivate later:**

```bash
deactivate
```

### Step 4: Install Dependencies

```bash
# Ensure pip is up to date
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

**What gets installed:**

- `imitation` - IRL/imitation learning algorithms
- `stable-baselines3` - Deep RL algorithms
- `gymnasium` - Environment interface
- `pettingzoo` - Multi-agent environments
- `pygame` - Environment rendering
- `wandb` - Experiment tracking
- `torch` - Deep learning framework
- `numpy`, `scipy` - Numerical computing

**Installation may take 5-10 minutes** depending on your internet connection.

### Step 5: Verify Installation

```bash
cd code/src

# Test imports
python3 << EOF
import gymnasium
import stable_baselines3
import imitation
import pygame
import wandb
print("✓ All packages imported successfully")
EOF

# Test environment registration
python3 -c "from common.common import register_environments; register_environments(); print('✓ Environments registered')"

# Test Pygame rendering
python3 -c "import pygame; pygame.init(); print('✓ Pygame initialized')"
```

---

## 🎮 GPU Setup

### NVIDIA GPU (Recommended for Training)

#### Step 1: Install NVIDIA Drivers

**Ubuntu:**

```bash
# Check current driver
nvidia-smi

# If not installed, install drivers
sudo apt update
sudo apt install nvidia-driver-525  # Or latest version
sudo reboot
```

**Windows/macOS:**
Download from [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)

#### Step 2: Install CUDA Toolkit

**Ubuntu:**

```bash
# Download CUDA 11.8 (example)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

**Verify CUDA:**

```bash
nvcc --version
```

#### Step 3: Install PyTorch with CUDA

```bash
# Activate virtual environment
source env/bin/activate

# Install PyTorch with CUDA 11.8 (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify GPU access in Python:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
# Should output: CUDA available: True
```

### Apple Silicon (M1/M2) Setup

```bash
# PyTorch with Metal acceleration
pip install torch torchvision torchaudio

# Verify
python3 -c "import torch; print(torch.backends.mps.is_available())"
# Should output: True
```

### CPU-Only Setup

If no GPU available, the default installation works but training will be slower:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## 🐳 Docker Setup

Docker provides reproducible environments, especially useful for GPU clusters.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU)

### Building Docker Image

```bash
cd docker

# Build image
docker build -t irl-intention-recognition:latest .

# Verify build
docker images | grep irl-intention-recognition
```

### Running Container (GPU)

```bash
# Run with GPU access
docker run --gpus all \
  -v $(pwd):/workspace \
  -it irl-intention-recognition:latest \
  /bin/bash

# Inside container, run experiments
cd /workspace/code/src
python3 run_main_experiment.py
```

### Running Container (CPU)

```bash
# Run without GPU
docker run \
  -v $(pwd):/workspace \
  -it irl-intention-recognition:latest \
  /bin/bash
```

### Pre-configured Experiment Scripts

```bash
# Inside container
cd /workspace/docker/start_container_commands

# Run baseline experiments
./run_baseline_experiments.sh

# Run main IRL experiments
./run_main_experiment.sh
```

---

## ✅ Verifying Installation

### Quick Verification

```bash
cd code/src

# Run quick test
python3 << EOF
import gymnasium as gym
from common.common import register_environments
from stable_baselines3 import PPO

register_environments()

# Test environment creation
env = gym.make('kaz-single-player-v0', type_of_player='knight')
print("✓ Environment created")

# Test quick training (very short)
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1000)
print("✓ Training works")

env.close()
print("\n✓✓✓ All systems operational! ✓✓✓")
EOF
```

### Comprehensive Test Suite

```bash
cd tests

# Basic environment tests
python3 test_classic_gym_game_set_up.py

# Environment-specific tests
python3 environment_tests/test_fixed_horizon_kaz_works.py
python3 environment_tests/test_variable_horizon_kaz_works.py

# Integration tests (takes longer)
python3 test_full_framework_functionality.py
```

### Visual Verification

```bash
cd tests/manual_gameplay

# Test with rendering (requires display)
python3 play_different_envs.py
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. `ModuleNotFoundError: No module named 'X'`

**Solution:**

```bash
# Ensure virtual environment is activated
source env/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# If specific package missing
pip install <package-name>
```

#### 2. `gym.error.UnregisteredEnv`

**Solution:**

```python
# Always register environments before use
from common.common import register_environments
register_environments()
```

#### 3. CUDA Out of Memory

**Solution:**

```yaml
# In configuration file, reduce:
experiment:
  n_envs: 2  # Reduce from 4 or 8

# Or force CPU
import torch
torch.cuda.is_available = lambda: False
```

#### 4. Pygame/Rendering Issues

**On Linux without display:**

```bash
# Use virtual display
sudo apt install xvfb
xvfb-run -a python3 your_script.py

# Or disable rendering
env = gym.make('kaz-single-player-v0', render_mode=None)
```

**On macOS:**

```bash
# Ensure XQuartz is running for remote displays
# Or use render_mode='rgb_array' instead of 'human'
```

#### 5. Slow Training

**Solutions:**

- Use GPU if available
- Increase `n_envs` for parallel environments
- Reduce `total_timesteps` for testing
- Use simpler neural network architectures

#### 6. Import Errors with Stable-Baselines3

**Solution:**

```bash
# Ensure compatible versions
pip install stable-baselines3==2.0.0
pip install gymnasium==0.28.1
```

#### 7. Weights & Biases Login

**Solution:**

```bash
# Login to W&B
wandb login

# Or disable W&B
export WANDB_MODE=disabled

# Or use offline mode
export WANDB_MODE=offline
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in project root:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your credentials
# Example .env file content:
WANDB_API_KEY=your_api_key_here
CUDA_VISIBLE_DEVICES=0  # Use GPU 0
OMP_NUM_THREADS=4       # CPU threads for parallel environments
```

**Security Note:** The `.env` file is in `.gitignore` and will never be committed. See [SECURITY.md](SECURITY.md) for more details.

### Weights & Biases Setup

```bash
# W&B is installed via requirements.txt
# Get your API key from: https://wandb.ai/authorize

# Login (one-time)
wandb login

# Test
python3 -c "import wandb; wandb.init(project='test'); print('✓ W&B configured')"
```

### Experiment Configuration

Main configuration files in `code/src/configuration/`:

**Baseline Experiments:**

```yaml
# kaz_baselines_config.yaml
experiment:
  n_envs: 4
  total_timesteps: 1000000
  seeds: [42, 123, 456]

environment:
  skill_level: "medium"
  horizon: "variable"
```

**IRL Experiments:**

```yaml
# main_experiment_config.yaml
experiment:
  n_envs: 4
  total_timesteps: 1000000

owner_agent:
  rl_algorithm: "PPO"

assistant_agent:
  rl_algorithm: "PPO"
  irl_algorithm: "AIRL"
```

---

## 🔧 Advanced Setup

### Multiple Python Versions

```bash
# Use specific Python version
python3.8 -m venv env38
source env38/bin/activate
pip install -r requirements.txt
```

### Development Tools

```bash
# Optional: Install development tools
pip install black flake8 pytest ipython jupyter

# Format code
black code/src/

# Lint code
flake8 code/src/

# Interactive Python
ipython
```

### Jupyter Notebook Setup

```bash
pip install jupyter notebook

# Create kernel for this project
python -m ipykernel install --user --name=irl-env --display-name "IRL Environment"

# Start Jupyter
jupyter notebook
```

---

## 📚 Next Steps

After successful installation:

1. **Read the Documentation**

   - Main [README.md](README.md)
   - [Code Documentation](code/src/README.md)
   - [Contributing Guide](CONTRIBUTING.md)

2. **Run Quick Experiments**

   ```bash
   cd code/src
   # Run small baseline test
   python3 run_kaz_baselines.py configuration/kaz_baselines_config.yaml
   ```

3. **Explore Examples**

   ```bash
   cd tests/manual_gameplay
   python3 play_different_envs.py
   ```

4. **Configure Your Experiments**

   - Edit YAML files in `code/src/configuration/`
   - Start with small `total_timesteps` for testing

5. **Track Your Experiments**
   - Set up Weights & Biases
   - Monitor training progress
   - Compare different algorithms

---

## 💬 Getting Help

If you encounter issues not covered here:

1. Check [Troubleshooting](#troubleshooting) section
2. Review [GitHub Issues](https://github.com/JoshWheeler08/Investigating-IRL-based-intention-recognition-algorithms/issues)
3. Open a new issue with detailed information
4. Include your system info, error messages, and steps to reproduce

---

**You're all set! Happy experimenting! 🚀**
