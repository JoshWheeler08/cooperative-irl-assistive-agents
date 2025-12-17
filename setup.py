"""
Setup script for IRL-based Intention Recognition package.
"""

from setuptools import setup, find_packages

setup(
    name="irl-intention-recognition",
    version="1.0.0",
    description="IRL-based Intention Recognition for Assistive Action Planning",
    author="Joshua Wheeler",
    author_email="",
    license="MIT",
    packages=find_packages(where="code/src"),
    package_dir={"": "code/src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "stable-baselines3>=1.7.0",
        "imitation>=0.4.0",
        "gymnasium>=0.28.0",
        "pettingzoo>=1.22.0",
        "pygame>=2.1.0",
        "wandb>=0.12.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ]
    },
)
