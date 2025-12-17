"""

    Simple program testing that my fixed-horizon KAZ environment works before including it in the experiment pipeline. 

"""

import os
import sys



import os
import sys



from stable_baselines3 import PPO
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import gym

# Register fixed horizon SingleRLPolicyDoubleKAZ environment

register(
    id="FixedHorizonSingleRLPolicyDoubleKAZ-v0",
    entry_point="environments.kaz_variants.fixed_horizon.fixed_horizon_single_policy_double_player:FixedHorizonSingleRLPolicyDoubleKAZ",
    max_episode_steps=None,
)


# Set hyperparameters
horizon_length = 1000


# Train PPO Algorithm for owner using SB3

env_kwargs = {
    "fixed_horizon_num_steps": horizon_length,
}

env = make_vec_env(
    "FixedHorizonSingleRLPolicyDoubleKAZ-v0", 
    n_envs=4, 
    env_kwargs=env_kwargs
) # Parallel environments

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000)


# Play Game with learned policy

env = gym.make(
    "FixedHorizonSingleRLPolicyDoubleKAZ-v0",
    **env_kwargs, 
    render_mode="human", 
    running_in_gym_flag=True
)

obs, info = env.reset()

total_timesteps = 1000
for i in range(total_timesteps):

    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    if(done):
        obs, info = env.reset()

env.close()
