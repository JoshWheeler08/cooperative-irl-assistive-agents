# 📚 Documentation Index

Welcome! This document serves as a navigation guide to all documentation in this repository.

## 🎯 Start Here

### New to the Project?

1. **[README.md](README.md)** - Project overview, features, and quick start
2. **[SETUP.md](SETUP.md)** - Detailed installation instructions
3. **[QUICKREF.md](QUICKREF.md)** - Quick reference for common tasks

### Ready to Contribute?

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project

### Want to Cite This Work?

- **[CITATION.cff](CITATION.cff)** - Citation information in CFF format
- **[LICENSE](LICENSE)** - MIT License and third-party licenses

---

## 📖 Core Documentation

### Main README

**[README.md](README.md)**

- Project overview and motivation
- Architecture diagram
- Quick start guide
- Key features and contributions
- Installation instructions
- Running experiments
- Project structure
- Technologies used
- Citation information

### Setup Guide

**[SETUP.md](SETUP.md)**

- System requirements
- Step-by-step installation
- GPU/CUDA setup
- Docker configuration
- Troubleshooting
- Environment variables
- Verification steps

### Quick Reference

**[QUICKREF.md](QUICKREF.md)**

- Common commands
- Configuration cheat sheet
- Algorithm comparison
- Quick fixes
- File locations

---

## 🔧 Technical Documentation

### Source Code Documentation

**[code/src/README.md](code/src/README.md)**

- Module descriptions
- Core experiment files
- Configuration guide
- Directory structure
- Extending the framework

### Environment Documentation

**[code/src/environments/README.md](code/src/environments/README.md)**

- KAZ game overview
- Environment variants
- Single vs. double player
- Skill levels
- Custom environments

### IRL Algorithms

**[code/src/irl_training/README.md](code/src/irl_training/README.md)**

- AIRL, GAIL, BC, DAgger, etc.
- Algorithm comparison
- Usage examples
- Implementation details
- Research insights

### Agent Architecture

**[code/src/agents/README.md](code/src/agents/README.md)**

- Owner agent (human)
- Assistant agent (robot)
- Architecture overview
- Training pipeline
- EnvObject and PolicyObject

---

## 🧪 Testing Documentation

### Test Suite Overview

**[tests/README.md](tests/README.md)**

- Unit tests
- Integration tests
- Environment tests
- Manual gameplay tests
- Example models
- Debugging tips

---

## 🤝 Contributing

### Contributing Guide

**[CONTRIBUTING.md](CONTRIBUTING.md)**

- Ways to contribute
- Development setup
- Adding new algorithms
- Adding new environments
- Code style
- Pull request process
- Bug reports
- Feature requests

---

## 📄 Legal & Citation

### License

**[LICENSE](LICENSE)**

- MIT License
- Third-party licenses
- Usage terms

### Citation Information

**[CITATION.cff](CITATION.cff)**

- BibTeX citation
- Thesis information
- Author details
- Keywords

---

## 📂 Documentation by Topic

### Installation & Setup

1. [README.md](README.md) - Quick Start section
2. [SETUP.md](SETUP.md) - Complete guide
3. [QUICKREF.md](QUICKREF.md) - Commands

### Configuration

1. [code/src/README.md](code/src/README.md) - Configuration section
2. [QUICKREF.md](QUICKREF.md) - Cheat sheet
3. Configuration files in `code/src/configuration/`

### Algorithms

1. [code/src/irl_training/README.md](code/src/irl_training/README.md) - IRL algorithms
2. [README.md](README.md) - Technologies section
3. [QUICKREF.md](QUICKREF.md) - Algorithm table

### Environments

1. [code/src/environments/README.md](code/src/environments/README.md) - Complete guide
2. [README.md](README.md) - Architecture section

### Testing

1. [tests/README.md](tests/README.md) - Test documentation
2. [CONTRIBUTING.md](CONTRIBUTING.md) - Testing guidelines

### Development

1. [CONTRIBUTING.md](CONTRIBUTING.md) - Full guide
2. [SETUP.md](SETUP.md) - Development tools
3. [code/src/README.md](code/src/README.md) - Extending framework

---

## 🗺️ Documentation Map

```
Repository Root
│
├── README.md                    ⭐ START HERE
├── SETUP.md                     🔧 Installation
├── QUICKREF.md                  ⚡ Quick reference
├── CONTRIBUTING.md              🤝 How to contribute
├── LICENSE                      ⚖️ Legal
├── CITATION.cff                 📖 How to cite
│
├── code/src/
│   ├── README.md               📚 Source code guide
│   │
│   ├── environments/
│   │   └── README.md           🎮 Environments
│   │
│   ├── irl_training/
│   │   └── README.md           🧠 IRL algorithms
│   │
│   └── agents/
│       └── README.md           🤖 Agent architecture
│
└── tests/
    └── README.md               🧪 Testing guide
```

---

## 📊 Documentation Status

| Document                        | Status      | Last Updated |
| ------------------------------- | ----------- | ------------ |
| README.md                       | ✅ Complete | Dec 2024     |
| SETUP.md                        | ✅ Complete | Dec 2024     |
| QUICKREF.md                     | ✅ Complete | Dec 2024     |
| CONTRIBUTING.md                 | ✅ Complete | Dec 2024     |
| code/src/README.md              | ✅ Complete | Dec 2024     |
| code/src/environments/README.md | ✅ Complete | Dec 2024     |
| code/src/irl_training/README.md | ✅ Complete | Dec 2024     |
| code/src/agents/README.md       | ✅ Complete | Dec 2024     |
| tests/README.md                 | ✅ Complete | Dec 2024     |
| LICENSE                         | ✅ Complete | Dec 2024     |
| CITATION.cff                    | ✅ Complete | Dec 2024     |

---

## 🔍 Finding What You Need

### I want to...

**...understand what this project does**
→ [README.md](README.md)

**...install and run experiments**
→ [SETUP.md](SETUP.md) → [QUICKREF.md](QUICKREF.md)

**...understand the code structure**
→ [code/src/README.md](code/src/README.md)

**...learn about IRL algorithms**
→ [code/src/irl_training/README.md](code/src/irl_training/README.md)

**...understand the game environments**
→ [code/src/environments/README.md](code/src/environments/README.md)

**...add a new algorithm**
→ [CONTRIBUTING.md](CONTRIBUTING.md) - Adding IRL Algorithm section

**...test my changes**
→ [tests/README.md](tests/README.md)

**...cite this work**
→ [CITATION.cff](CITATION.cff) or [README.md](README.md) - Citation section

**...report a bug**
→ [CONTRIBUTING.md](CONTRIBUTING.md) - Bug Reports section

**...understand the agent architecture**
→ [code/src/MyAgents/README.md](code/src/MyAgents/README.md)

---

## 💡 Documentation Tips

### For Beginners

1. Start with main README
2. Follow SETUP guide
3. Run baseline experiments
4. Explore examples in tests/manual_gameplay
5. Read module documentation as needed

### For Researchers

1. Review README for research context
2. Read IRL algorithms documentation
3. Understand environment variants
4. Configure experiments via YAML files
5. Track experiments with W&B

### For Developers

1. Read CONTRIBUTING guide
2. Understand code structure
3. Review module documentation
4. Write tests for changes
5. Follow code style guidelines

---

## 📮 Feedback on Documentation

Found an issue with documentation?

- **Typo/Error**: Open an issue on GitHub
- **Missing Info**: Open an issue describing what's needed
- **Improvement Idea**: Open a discussion or issue
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 🔄 Documentation Versioning

This documentation corresponds to:

- **Repository Version**: 1.0.0
- **Last Updated**: December 2024
- **Python Version**: 3.8+
- **Major Dependencies**:
  - Stable-Baselines3
  - Imitation
  - PettingZoo
  - Gymnasium

For older versions, check git tags and corresponding documentation.

---

## 📚 External Resources

### Research Papers

Referenced in [code/src/irl_training/README.md](code/src/irl_training/README.md)

### Library Documentation

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Imitation](https://imitation.readthedocs.io/)
- [PettingZoo](https://pettingzoo.farama.org/)
- [Gymnasium](https://gymnasium.farama.org/)

### Related Projects

- [OpenAI Gym](https://www.gymlibrary.dev/)
- [Weights & Biases](https://docs.wandb.ai/)

---

**Happy Learning! 📖**

If you can't find what you're looking for, please open an issue on GitHub.
