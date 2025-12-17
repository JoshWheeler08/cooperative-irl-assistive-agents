import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

import numpy as np

from irl_training.irl_base import IRLImplementation

class MyGAIL(IRLImplementation):

    """ 
        GAIL implementation adapted from example code provided by the Imitation library 
        to fit my IRL template.

        (https://imitation.readthedocs.io/en/latest/algorithms/gail.html) 
    """

    def __init__(self, owner_agent, env_obj, training_args=None):

        if training_args == None: # no irl arguments passed
            # use default parameters
            training_args = {
                "min_episodes": 60,
                "total_timesteps": 20000,
                "PPO_n_epochs":10,
            }

        super().__init__(owner_agent, env_obj, training_args)

    def train(self):
        # Output IRL configuration information
        self.output_config_info()

        # Create environment
        env = gym.make(self.env_obj.env_id, **self.env_obj.env_kwargs)

        rng = np.random.default_rng(self.env_obj.seed)

        expert = self.owner_agent.policy

        # Generate some expert trajectories that the discriminator needs to distinguish from the approximated policy's trajectories
        rollouts = rollout.rollout(
            expert,
            make_vec_env(
                self.env_obj.env_id,
                n_envs=self.env_obj.n_envs,
                post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
                env_make_kwargs=self.env_obj.env_kwargs,
                rng=rng,
            ),
            rollout.make_sample_until(min_timesteps=None, min_episodes=self.args["min_episodes"]),
            rng=rng,
        )

        # Set up GAIL Trainer

        venv = make_vec_env(
                self.env_obj.env_id, 
                n_envs=self.env_obj.n_envs, 
                env_make_kwargs=self.env_obj.env_kwargs,
                rng=rng,
            )

        learner = PPO( # default PPO values
            env=venv,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=self.args["PPO_n_epochs"],
        )

        reward_net = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
        )

        gail_trainer = GAIL(
            demonstrations=rollouts,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
            custom_logger=self.env_obj.logger,
        )

        # Train Model using GAIL
        gail_trainer.train(self.args["total_timesteps"])  # Note: set to 300000 for better results -> Imitation library comment

        # Tidy up
        env.close()

        # Return approximated owner agent policy
        return learner.policy