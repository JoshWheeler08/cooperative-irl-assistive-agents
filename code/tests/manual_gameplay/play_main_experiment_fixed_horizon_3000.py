"""
    This is a small program which loads the agents trained during a main experiment test investigating the performance impact of 
    using a *fixed-horizon* version of KAZ set to 3000 timesteps. 
        
"""

import os
import sys
import gym



from gym.envs.registration import register
from wrappers.kaz_training_wrapper import KAZTrainingWrapper
from agents.helpers.env_config import EnvObject
from agents.assistant import Assistant
from agents.owner import Owner

from common.constants import IGNORE_HORIZON
from common.common import select_skill_level

# Register Environment

register(
    id="DoubleKAZ-v0",
    entry_point="environments.kaz.core.fixed_horizon_double_player:FixedHorizonDoubleKAZ",
    max_episode_steps=None,
)

# Set Skill Level
select_skill_level('vanilla')


# Load models
assistant_agent = Assistant()
path = "./example_models/main_experiment_fixed_horizon_3000/assistant/"
assistant_agent.load_policy(path, final_policy_type="PPO", rl_policy_type="PPO", irl_alg="Density")

owner_agent = Owner()
path = path = "./example_models/main_experiment_fixed_horizon_3000/owner/"
owner_agent.load_policy(path, policy_type="PPO")


# Play Game

env_kwargs = {
    "knight_policy": owner_agent.policy, # controls knight's actions in environment 
    "env_output":False,
    "fixed_horizon_num_steps": 1500,
    "max_zombies": 10,
}

# Object describing the KAZ environment to be used
env_obj = EnvObject(
    env_id="DoubleKAZ-v0",
    n_envs=1,
    total_timesteps=10000,
    env_kwargs=env_kwargs,
    wrapper_class=KAZTrainingWrapper,
    seed=100,
)


assistant_agent.play_game_with_final_policy(env_obj=env_obj)