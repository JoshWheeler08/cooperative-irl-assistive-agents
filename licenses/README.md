# Third-Party Licenses

This directory contains the licenses for third-party libraries used in this project.

## Included Libraries

### [Imitation](imitation-license)

**License**: MIT License  
**Purpose**: Provides implementations of imitation learning algorithms (AIRL, GAIL, BC, DAgger, etc.)  
**Repository**: https://github.com/HumanCompatibleAI/imitation

### [OpenAI Gym](open-ai-gym-license)

**License**: MIT License  
**Purpose**: Provides the foundational RL environment interface  
**Repository**: https://github.com/openai/gym  
**Note**: This project uses Gymnasium (maintained fork)

### [PettingZoo](petting-zoo-license)

**License**: MIT License  
**Purpose**: Multi-agent environment interface and utilities  
**Repository**: https://github.com/Farama-Foundation/PettingZoo

### [Stable-Baselines3](stable-baselines3-license)

**License**: MIT License  
**Purpose**: Provides high-quality implementations of RL algorithms (PPO, A2C, DQN)  
**Repository**: https://github.com/DLR-RM/stable-baselines3

## Other Dependencies

The following dependencies are used but not included here (check PyPI for their licenses):

- **PyTorch**: BSD-style license
- **NumPy**: BSD license
- **Pygame**: LGPL license
- **Weights & Biases**: Proprietary (free tier available)
- **PyYAML**: MIT license

## Compliance

This project complies with all third-party licenses. All included libraries use permissive open-source licenses (MIT, BSD) that allow for academic and research use.

## Attribution

If you use this project, please also cite the original libraries:

```bibtex
@misc{gleave2022imitation,
  author = {Gleave, Adam and Taufeeque, Mohammad and Rocamonde, Juan and Jenner, Erik and Wang, Steven H. and Toyer, Sam and Ernestus, Maximilian and Belrose, Nora and Emmons, Scott and Russell, Stuart},
  title = {imitation: Clean Imitation Learning Implementations},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HumanCompatibleAI/imitation}},
}

@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

---

**Note**: This project's main code is licensed under MIT License (see root LICENSE file).
