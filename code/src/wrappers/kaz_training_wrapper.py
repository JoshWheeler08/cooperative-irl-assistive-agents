"""
OpenAI Gym Wrapper for augmenting KAZ environment observations.

This wrapper augments vector-state observations (not pixel-based) by adding
predictions from the RL policy and internal model policy to the Assistant agent's observations.
"""

import numpy as np
import gym
from gym import spaces


class KAZTrainingWrapper(gym.Wrapper):
    """
    OpenAI Gym Wrapper for augmenting the observation returned by the environment 
    before giving it to the Assistant agent.
    
    This wrapper is only used for augmenting vector-state observations, not pixel-based ones.
    See DoubleKAZ - observe() for observation details.
    """

    def __init__(self, env, rl_policy, internal_model_policy, running_in_gym_flag=False):
        """
        Initialize the KAZ Training Wrapper.
        
        Args:
            env: The base KAZ environment to wrap
            rl_policy: The RL policy for predicting actions on archer observations
            internal_model_policy: The internal model policy for predicting owner/knight actions
            running_in_gym_flag: Flag indicating if running in newer Gym API (with info dict on reset)
        """
        super().__init__(env)
        self.rl_policy = rl_policy
        self.internal_model_policy = internal_model_policy
        self.running_in_gym_flag = running_in_gym_flag
        
        # Final Assistant policy is trained on observations of the entire game, 
        # rather than having a local scope, so need to update num_tracked formula 
        self.num_tracked = (
            self.env.num_archers + self.env.num_knights + self.env.max_zombies + 
            self.env.num_knights + self.env.max_arrows
        )

        # Modifying the observation space to include the two extra pieces of information that are being appended 
        # (rl_policy(Archer_Observation), internal_model_policy(Knight_Observation))
        shape = (
            [512, 512, 3]
            if not self.vector_state
            else [1, ((self.num_tracked + 1) * (self.vector_width + 1)) + 2]  # + 2 for the extra values
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        
        # Updating observation space
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

    def _update_observation(self, obs):
        """
        Adds the RL policy and internal model policy predictions to the observation.
        
        Args:
            obs: Original observation from the environment
            
        Returns:
            Modified observation with RL and internal model predictions appended
        """
        # Get RL policy prediction on current observation
        rl_action, _states = self.rl_policy.predict(obs)

        # Get owner/knight observation
        owner_obs = self.env.get_knight_observation()

        # Get internal model prediction on owner observation
        internal_action, _states = self.internal_model_policy.predict(owner_obs)

        # Combine the two predictions to make a new observation
        full_game_obs = self.env.get_full_observation()  # full game observation from the perspective of assistant

        # Append policy predictions to observation
        new_obs = np.append(full_game_obs, [rl_action, internal_action])

        # Output has to be flattened
        new_obs = new_obs.reshape([1, len(new_obs)])

        return new_obs

    def step(self, action):
        """
        Execute one step in the environment with augmented observations.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (augmented_observation, reward, done, info)
        """
        obs, reward, done, info = self.env.step(action)

        # Modify environment output by augmenting observation
        new_obs = self._update_observation(obs)

        return new_obs, reward, done, info

    def reset(self):
        """
        Reset the environment and return augmented initial observation.
        
        Returns:
            Augmented initial observation (and info dict if running_in_gym_flag is True)
        """
        if self.running_in_gym_flag:
            obs, info = self.env.reset()

            # Modify environment output by augmenting observation
            new_obs = self._update_observation(obs)

            return new_obs, info

        else:
            obs = self.env.reset()

            # Modify environment output by augmenting observation
            new_obs = self._update_observation(obs)

            return new_obs
