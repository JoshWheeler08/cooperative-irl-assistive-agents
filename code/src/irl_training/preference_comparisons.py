
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.ppo import MlpPolicy

from irl_training.irl_base import IRLImplementation

class MyPrefComp(IRLImplementation):

    """ 
        Preference Comparisons implementation adapted from example code provided by the Imitation library 
        to fit my IRL template.

        (https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html) 
    """

    def __init__(self, owner_agent, env_obj, training_args=None):
        
        if training_args == None: # no irl arguments passed
            # use default parameters
            training_args = {
                "reward_trainer_epochs": 3,
                "pref_comp_num_iterations": 5,
                "total_timesteps": 5000,
                "total_comparisons": 200,
                "PPO_n_epochs": 10,
                "PPO_learn_total_timesteps": 1000,
            }
        
        super().__init__(owner_agent, env_obj, training_args)


    def train(self):
        """ Preference Comparison algorithms learn a reward function by comparing trajectory segments to each other """
        
        # Output IRL configuration information
        self.output_config_info()

        rng = np.random.default_rng(self.env_obj.seed)

        # Set up environment
        venv = make_vec_env(
            env_name=self.env_obj.env_id, 
            n_envs=self.env_obj.n_envs, 
            env_make_kwargs=self.env_obj.env_kwargs,
            rng=rng,
        )

        # Set up reward net
        reward_net = BasicRewardNet(
            venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
        )

        # Fragmenter creates pairs of trajectory fragments from sets of trajectories
        fragmenter = preference_comparisons.RandomFragmenter(
            warning_threshold=0,
            rng=rng,
        )

        # Computes synthetic preferences using ground-truth environment rewards
        gatherer = preference_comparisons.SyntheticGatherer(rng=rng)

        # Converts two fragments' rewards into preference probability
        preference_model = preference_comparisons.PreferenceModel(reward_net)

        # Trains the reward model
        reward_trainer = preference_comparisons.BasicRewardTrainer(
            preference_model=preference_model,
            loss=preference_comparisons.CrossEntropyRewardLoss(),
            epochs=self.args["reward_trainer_epochs"],
            rng=rng,
        )

        # Get the expert
        expert = self.owner_agent.policy

        # Trains SB3 Agent on reward function
        trajectory_generator = preference_comparisons.AgentTrainer(
            algorithm=expert,
            reward_fn=reward_net,
            venv=venv,
            exploration_frac=0.0,
            rng=rng,
        )

        pref_comparisons = preference_comparisons.PreferenceComparisons(
            trajectory_generator,
            reward_net,
            num_iterations=self.args["pref_comp_num_iterations"],
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            reward_trainer=reward_trainer,
            fragment_length=100,
            transition_oversampling=1,
            initial_comparison_frac=0.1,
            allow_variable_horizon=True,
            initial_epoch_multiplier=1,
            custom_logger=self.env_obj.logger,
        )

        # Start Training
        pref_comparisons.train(
            total_timesteps=self.args["total_timesteps"],  # For good performance this should be 1_000_000
            total_comparisons=self.args["total_comparisons"],  # For good performance this should be 5_000
        )

        # After training the reward function, we can wrap our environment with it and train an SB3 Agent - the agent only sees these rewards

        learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)

        # SB3 Agent
        learner = PPO(
            policy=MlpPolicy,
            env=learned_reward_venv,
            seed=self.env_obj.seed,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=self.args["PPO_n_epochs"],
            n_steps=64,
        )

        learner.learn(self.args["PPO_learn_total_timesteps"])  # Note: set to 100000 to train a proficient expert

        # Tidy up
        venv.close()

        # Return approximated Owner policy
        return learner.policy
