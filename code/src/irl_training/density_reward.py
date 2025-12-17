import pprint

from imitation.algorithms import density as db
from imitation.data import types
from imitation.util import util

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO
from imitation.data import rollout
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data.wrappers import RolloutInfoWrapper
import gym
import numpy as np

from irl_training.irl_base import IRLImplementation

class MyDensity(IRLImplementation):

    """ 
        (Kernel) Density-based reward modelling implementation adapted from example code provided by the Imitation library 
        to fit my IRL template.

        (https://imitation.readthedocs.io/en/latest/algorithms/density.html) 
    """

    def __init__(self, owner_agent, env_obj, training_args=None):

        if training_args == None: # no irl arguments passed
            # use default parameters
            training_args = {
                "eval_num_trajectories": 1,
                "fast" : False,
                "fast_n_vec": 4,
                "fast_n_trajectories": 100,
                "fast_n_iterations": 1000,
                "fast_n_rl_train_steps": int(1e5),
                "min_episodes": 57,
                "min_timesteps": 100,
            }

        super().__init__(owner_agent, env_obj, training_args)


    def train(self):
        """ Density Based Reward Modelling """

        # Output IRL configuration information
        self.output_config_info()

        # Set FAST = False for longer training. Use True for testing and CI.
        FAST = self.args['fast']

        if FAST:
            # Just testing it works, so the values don't really matter
            N_VEC = 1
            N_TRAJECTORIES = 1
            N_ITERATIONS = 1
            N_RL_TRAIN_STEPS = 10

        else:
            N_VEC = self.args['fast_n_vec']
            N_TRAJECTORIES = self.args['fast_n_trajectories']
            N_ITERATIONS = self.args['fast_n_iterations']
            N_RL_TRAIN_STEPS = self.args['fast_n_rl_train_steps']
        

        # Set up environment 
        rng = np.random.default_rng(self.env_obj.seed)
        env_name = self.env_obj.env_id
        expert = self.owner_agent.policy
        rollout_env = DummyVecEnv(
            [lambda: RolloutInfoWrapper(gym.make(self.env_obj.env_id, **self.env_obj.env_kwargs)) for _ in range(N_VEC)]
        )

        # Collect rollouts/trajectories using the expert agent in the specified environment 
        rollouts = rollout.rollout(
            expert,
            rollout_env,
            rollout.make_sample_until(min_timesteps=self.args["min_timesteps"], min_episodes=self.args["min_episodes"]),
            rng=rng,
        )

        env = util.make_vec_env(env_name, n_envs=N_VEC, env_make_kwargs=self.env_obj.env_kwargs, rng=rng)

        # Train Agent using rollouts/demonstrations
        imitation_trainer = PPO(ActorCriticPolicy, env, learning_rate=3e-4, n_steps=2048)
        
        density_trainer = db.DensityAlgorithm(
            venv=env,
            demonstrations=rollouts,
            rl_algo=imitation_trainer,
            density_type=db.DensityType.STATE_ACTION_DENSITY,
            is_stationary=True,
            kernel="gaussian",
            kernel_bandwidth=0.2,  # found using divination & some palm reading
            standardise_inputs=True,
            allow_variable_horizon=True,
            custom_logger=self.env_obj.logger,
            rng=rng,
        )

        density_trainer.train()
        
        # Return approximated Owner policy
        return density_trainer.policy


    # Outputs the results after evaluating the learned policy
    def _print_stats(self, density_trainer, n_trajectories, epoch=""):
        """ Outputs the results after testing the policy """

        # test learned policy using ground truth reward function from underlying environment 
        stats = density_trainer.test_policy(
            n_trajectories=n_trajectories,
            true_reward=True,
        )
        print("True reward function stats:")
        pprint.pprint(stats)

        # test learned policy using approximated (imitation) reward function
        stats_im = density_trainer.test_policy(
            true_reward=False, 
            n_trajectories=n_trajectories,
        )

        print(f"Imitation reward function stats, epoch {epoch}:")
        pprint.pprint(stats_im)


    # Evaluates the learned policy 
    def evaluate(self):
        if self.policy == None:
            print("[Error] Need to train policy first - aborting")
            return

        elif self.env == None:
            print("[Error] Evaluation env can't be of type None - aborting")
            return

        else:
            print("\n[Starting] Evaluation:\n")

            self._print_stats(
                density_trainer = self.policy, 
                n_trajectories = self.args["eval_num_trajectories"],
            )

            print("\n[Complete]\n")