"""

    Simple program testing that my variable-horizon KAZ environment works before including it in the experiment pipeline. 

"""

import os
import sys



from stable_baselines3 import PPO
from gym.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.evaluation import evaluate_policy

from wrappers.kaz_training_wrapper import KAZTrainingWrapper


# Register variable-horizon KAZ environments for training Knight and Archer agents

register(
    id="SingleKAZ-v0",
    entry_point="environments.kaz_core.single_player:SingleKAZ",
    max_episode_steps=None,
)

register(
    id="DoubleKAZ-v0",
    entry_point="environments.kaz_core.double_player:DoubleKAZ",
    max_episode_steps=None,
)


# Train PPO Algorithm for Owner/Human player
env_kwargs = {
    "type_of_player":"knight"
}

# Create env
env = make_vec_env("SingleKAZ-v0", n_envs=4, env_kwargs=env_kwargs) # Parallel environments

model_knight = PPO("MlpPolicy", env, verbose=1) # Using SB3 PPO implementation 

model_knight.learn(total_timesteps=100)


# Train PPO Algorithm for Archer - learning Archer's RL policy 

env_kwargs = {
    "type_of_player":"archer"
}

env = make_vec_env("SingleKAZ-v0", n_envs=4, env_kwargs=env_kwargs) # Parallel environments

model_archer = PPO("MlpPolicy", env, verbose=1)

model_archer.learn(total_timesteps=100)


# Train the Assistant's Final policy on Double-player KAZ by combining its learned RL policy with an internal model of the knight, 
# which in this case is the ground-truth model.

env_kwargs = {
    "knight_policy":model_knight,
}


wrapper_kwargs = {
    "rl_policy":model_archer,
    "internal_model_policy": model_knight,
} # In this case the internal model perfectly models the Owner player/Knight

env = make_vec_env(env_id="DoubleKAZ-v0", wrapper_class=KAZTrainingWrapper, env_kwargs=env_kwargs, wrapper_kwargs=wrapper_kwargs, n_envs=4) # Parallel environments

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100)


# Evaluate Final Policy with Knight in double-player KAZ
mean_reward, std_reward = evaluate_policy(model, env, 100)
print(mean_reward, std_reward)