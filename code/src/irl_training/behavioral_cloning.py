import gym
import time
import numpy as np

import torch as th

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from irl_training.irl_base import IRLImplementation

class MyBC(IRLImplementation):

    """ 
        Behaviour Cloning implementation adapted from example code provided by the Imitation library to fit my IRL template.

        (https://imitation.readthedocs.io/en/latest/algorithms/bc.html) 
    """

    def __init__(self, owner_agent, env_obj, training_args=None):

        if training_args == None: # no irl arguments passed
            # use default parameters
            training_args = {
                "min_episodes": 500,
                "n_epochs": 1000,
            }

        super().__init__(owner_agent, env_obj, training_args)


    def _sample_trajectories(self, owner_agent, env, rng):
        
        # Use Owner Agent to sample trajectories
        rollouts = rollout.rollout( # Collecting 50 episode rollouts
            owner_agent.policy,
            DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
            rollout.make_sample_until(min_timesteps=None, min_episodes=self.args["min_episodes"]),
            rng=rng,
        )

        # we flatten the trajectories because behaviour cloning is only interested in individual transitions
        transitions = rollout.flatten_trajectories(rollouts) 

        #print(
        #    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
        #    After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
        #    The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
        #    """
        #)

        return transitions


    # Work modified from https://imitation.readthedocs.io/en/latest/tutorials/1_train_bc.html
    def train(self):

        # Output IRL configuration information
        self.output_config_info()

        rng = np.random.default_rng(self.env_obj.seed)

        # Create environment for generating transitions
        env = gym.make(self.env_obj.env_id, **self.env_obj.env_kwargs)


        # Get transitions for training by sampling trajectories
        transitions = self._sample_trajectories(self.owner_agent, env, rng)


        # Set up behaviour cloning algorithm 

        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=transitions,
            custom_logger=self.env_obj.logger,
            rng=rng,
        ) # using the transitions collected from the rollout!


        # Train policy
        bc_trainer.train(n_epochs=self.args["n_epochs"])

        # Tidy up
        env.close()

        # return IRL-learned policy object, which is the approximated Owner policy
        return bc_trainer