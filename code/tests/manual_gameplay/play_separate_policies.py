"""
    This is a small program which loads the agents trained during the baseline test `test_separate_policies`, 
    and plays them on the *DoubleKAZ* environment (variable time horizon).
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
    entry_point="environments.kaz.core.double_player:DoubleKAZ",
    max_episode_steps=None,
)

# Set Skill Level
select_skill_level('vanilla')


# Load models - in the baseline tests, the assistant agent doesn't support intention recognition, 
# so it is easier to use an Owner agent object to load it
owner_agent = Owner()
assistant_agent = Owner()


assistant_path = "./example_models/separate_policies/archer_"
owner_path ="./example_models/separate_policies/knight_"

owner_agent.load_policy(owner_path, policy_type="PPO")
assistant_agent.load_policy(assistant_path, policy_type="PPO")

# Play Game

env_kwargs = {
    "knight_policy": owner_agent.policy, # owner policy will control knight's actions
    "env_output":False,
    "fixed_horizon_num_steps": -1,
    "max_zombies": 10,
}

# Object to describe desired game environment
env_obj = EnvObject(
    env_id="DoubleKAZ-v0",
    n_envs=1,
    total_timesteps=10000,
    env_kwargs=env_kwargs,
    seed=100,
)


assistant_agent.play_game_with_rl_policy(env_obj)