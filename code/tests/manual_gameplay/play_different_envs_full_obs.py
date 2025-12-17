"""
    This is a small program which loads the agents trained during the baseline test 
    `test_train_on_different_envs_full_observation`, and plays them on the *FullObsDoubleKAZ* environment 
    (variable time horizon).
    
"""

import os
import sys
import gym

# Add src directory to path so imports work
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from gym.envs.registration import register
from wrappers.kaz_training_wrapper import KAZTrainingWrapper
from agents.helpers.env_config import EnvObject
from agents.assistant import Assistant
from agents.owner import Owner

from common.constants import IGNORE_HORIZON
from common.common import select_skill_level


# Register Environment

register(
    id="FullObsDoubleKAZ-v0",
    entry_point="environments.kaz.variants.variable_horizon.full_obs_double_player:FullObsDoubleKAZ",
    max_episode_steps=None,
)

# Set Skill Level
select_skill_level('vanilla')


# Load models - in the baseline tests, the assistant agent doesn't include intention recognition, 
# so it is easier to use an Owner agent object to load it
owner_agent = Owner()
assistant_agent = Owner() 

assistant_path = "./example_models/different_envs_full_obs/archer_"
owner_path ="./example_models/different_envs_full_obs/knight_"

owner_agent.load_policy(owner_path, policy_type="PPO")
assistant_agent.load_policy(assistant_path, policy_type="PPO")

# Play Game

env_kwargs = {
    "knight_policy": owner_agent.policy, # owner policy will determine the knight's next action per training step
    "env_output":False, 
    "fixed_horizon_num_steps": -1, # not using a fixed horizon variant of KAZ
    "max_zombies": 10, # number of zombies per wave
}

# EnvObject for describing the KAZ environment to be used
env_obj = EnvObject(
    env_id="FullObsDoubleKAZ-v0",
    n_envs=1,
    total_timesteps=10000,
    env_kwargs=env_kwargs,
    seed=100,
)

assistant_agent.play_game_with_rl_policy(env_obj)