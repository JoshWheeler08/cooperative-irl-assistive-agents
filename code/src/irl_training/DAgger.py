import gym

import tempfile
import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from stable_baselines3.common.evaluation import evaluate_policy

from irl_training.irl_base import IRLImplementation

class MyDAgger(IRLImplementation):
    
    """ 
        DAgger (Dataset Aggregation) implementation adapted from example code provided by the Imitation library 
        to fit my IRL template.

        (https://imitation.readthedocs.io/en/latest/algorithms/dagger.html) 
    """

    def __init__(self, owner_agent, env_obj, training_args=None):

        if training_args == None: # no irl arguments passed
            # use default parameters
            training_args = {
                "total_timesteps": 2000,
                "prefix" : "dagger_example_"
            }

        super().__init__(owner_agent, env_obj, training_args)


    def train(self):
        
        # DAgger is an extension of the behaviour cloning algorithm

        # Output IRL configuration information
        self.output_config_info()

        rng = np.random.default_rng(self.env_obj.seed)

        expert = self.owner_agent.policy

        # Create environment
        env = gym.make(self.env_obj.env_id, **self.env_obj.env_kwargs)

        venv = DummyVecEnv([lambda: gym.make(self.env_obj.env_id, **self.env_obj.env_kwargs)])

        # Reuse BC algorithm
        bc_trainer = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            rng=rng,
        )

        with tempfile.TemporaryDirectory(prefix=self.args["prefix"]) as tmpdir:
            print(tmpdir)
            dagger_trainer = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=tmpdir,
                expert_policy=expert,
                bc_trainer=bc_trainer,
                custom_logger=self.env_obj.logger,
                rng=rng,
            )

            # Train Model
            dagger_trainer.train(self.args["total_timesteps"])

        # Tidy up
        env.close()

        # Return approximated Owner policy 
        return dagger_trainer.policy