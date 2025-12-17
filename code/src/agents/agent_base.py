import gym

from abc import ABC
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from common.common import process_evaluation_results

class Agent(ABC):

    """ Abstract Base Class for representing a RL agent """

    def __init__(self, name="Agent"):
        self.name = name
        self.policy = None
        self.supported_rl_algs = {
            "PPO":PPO, 
            "A2C":A2C, 
            "DQN":DQN
        }

    # Returns agent's policy
    def policy(self):
        return self.policy


    # Learns a RL policy 
    def train_rl(self,
            env_obj, 
            policy="PPO", 
            policy_type="MlpPolicy",
    ):
        
        if policy in self.supported_rl_algs: # Check RL Algorithm is valid

            if(env_obj == None or env_obj.env_id == None):
                print("[Error] No valid environment given train_rl() - aborting")
                return None

            try:
                env_obj.n_envs = env_obj.n_envs if env_obj.n_envs > 0 else 4 # Check n_envs
                
                # Add seed to env_kwargs
                env_obj.env_kwargs["user_seed"] = env_obj.seed

                # Make vectorised environment
                env = make_vec_env(
                    env_id=env_obj.env_id,
                    wrapper_class=env_obj.wrapper_class,
                    wrapper_kwargs=env_obj.wrapper_kwargs,
                    n_envs=env_obj.n_envs, 
                    env_kwargs=env_obj.env_kwargs,
                )
                
                # Load SB3 Model
                model = self.supported_rl_algs[policy](
                    policy=policy_type,
                    env=env,
                    verbose=1,
                    seed=env_obj.seed,
                )

                # Set Logger
                if env_obj.logger is not None:
                    model.set_logger(env_obj.logger)

                # Learn policy on vectorised environment 
                model.learn(
                    total_timesteps=int(env_obj.total_timesteps),
                )

                env.close()
                
                print("[SUCCESS] RL training successful")

                return model
            except Exception as e:
                print(f"[Error] Failed to create the environment (error message: {e})")
                return None


        else:
            print(f"[Error] Invalid policy - {policy}")
            return None

    # Evaluates a learned RL policy
    def evaluate_policy(
        self, 
        env_obj=None, 
        n_eval_episodes=10, 
        policy=None,
        csv_path="./",
    ):
        if(policy == None or env_obj == None or env_obj.env_id == None): # input sanitisation check
            print("[Error] - Need to train the policy first and/or give a valid env id")
        else:
            try:
                # Make Vectorised Environment
                env = make_vec_env(
                    env_id=env_obj.env_id, 
                    n_envs=env_obj.n_envs, 
                    env_kwargs=env_obj.env_kwargs,
                    wrapper_class=env_obj.wrapper_class,
                    wrapper_kwargs=env_obj.wrapper_kwargs,
                )

                # Evaluate Policy using SB3 Method
                episode_rewards, episode_lengths = evaluate_policy(
                    policy, 
                    env, 
                    n_eval_episodes,
                    return_episode_rewards=True,
                )

                # Process evaluation data
                mean_reward, std_reward, mean_episode_length, csv_file_name = process_evaluation_results(episode_rewards, episode_lengths, csv_path)

                return mean_reward, std_reward, mean_episode_length

            except Exception as e:
                print(f"[Error] Problem creating env (error message: {e})")
        return None

    # Saves a policy to the provided path
    def save_policy(self, path=None, policy=None):
        if(policy == None): # input check 
            print("[Error] Invalid Policy - aborting")
            return

        if(path == None):
            print(f"[Error] Cannot save {self.name} due to missing path")
            return 
        else:
            policy.save(path) # using SB3 save method
            print(f"Success, policy details saved at {path}")


    # Loads a policy from provided path 
    def load_policy(self, path=None, policy_type=None):
        if policy_type in self.supported_rl_algs and path != None: # input check 
            model = self.supported_rl_algs[policy_type].load(path) # using SB3 method
            print(f"Success, policy details loaded from {path}")
            return model
        else:
            print(f"[Error] Cannot load policy due to missing/invalid arguments")
            return None


    # Plays game with learned policy using OpenAI Gym's simulator
    def play_game_with_policy(self, env_obj=None, render_mode="human", policy=None):
        if(policy == None): # input check
            print("Need to pass in a valid policy")
            return

        try:
            # Make game environment
            env = gym.make(env_obj.env_id, **env_obj.env_kwargs, render_mode=render_mode, running_in_gym_flag=True)

            obs, info = env.reset()
            
            total_timesteps = env_obj.total_timesteps
            for i in range(total_timesteps):
                action, _states = policy.predict(obs) # get agent action 

                obs, rewards, done, info = env.step(action) # see result of agent action on environment 

                if(done):
                    obs, info = env.reset()

            env.close()

        except Exception as e:
            print(f"[Error] Issue creating environment for play_game_with_policy() (error message : {e})")