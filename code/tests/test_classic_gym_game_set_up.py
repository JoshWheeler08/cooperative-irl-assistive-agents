""" 
    Small program used to verify that my KAZ game environments (`./src/environments/`) can be rendered and played 
    before including any trained agents. As a result, the knight's and archer's actions per timestep 
    are randomly selected by sampling from their respective actions spaces.

"""

import os
import sys
import gym

# Add src directory to path so imports work
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gym.envs.registration import register
from common.common import select_skill_level


max_episode_steps = None  # No limit on episode length

# Register Environments

register(
    id="SingleKAZ-v0",
    entry_point="environments.kaz.core.fixed_horizon_single_player:FixedHorizonSingleKAZ",
    max_episode_steps=max_episode_steps,
)

register(
    id="DoubleKAZ-v0",
    entry_point="environments.kaz.core.fixed_horizon_double_player:FixedHorizonDoubleKAZ",
    max_episode_steps=max_episode_steps,
)


# Set skill level
select_skill_level("medium")


# Play Game

# Set up hyperparameters
env_id = "DoubleKAZ-v0"

env_kwargs = {
    "env_output": False,
    "user_seed": 500,
}
render_mode = "human"
total_timesteps = 10000

# Make game environment
env = gym.make(env_id, **env_kwargs, render_mode=render_mode, running_in_gym_flag=True)

obs, info = env.reset()

for i in range(total_timesteps):
    action = env.action_space.sample()  # pick a random action for the archer
    # the knight's action is randomly sampled internally by the environment because OpenAI Gym doesn't support a multi-agent API/interface

    obs, rewards, done, info = env.step(
        action
    )  # Perform action, get new observation of environment with reward

    if done:
        # if one of the agents has died, or a zombie has reached the other side of the screen
        # Reset game
        obs, info = env.reset()

env.close()
