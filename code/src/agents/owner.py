
from agents.agent_base import Agent

class Owner(Agent):

    """ Python Class for representing the Owner/Human Player """

    def __init__(self, name="Owner_Agent_1"):
        super().__init__(name=name)
    

    # Evaluates learned policy
    def evaluate_policy(self, env_obj=None, n_eval_episodes=10, csv_path="./"):
        return super().evaluate_policy(
            env_obj, 
            n_eval_episodes, 
            self.policy,
            csv_path=csv_path,
        )


    # Saves policy at given path
    def save_policy(self, path=None, policy_type=""):
        if(path == None):
            print("[Error] Invalid path - aborting")
            return 

        path = f"{path}policy_{policy_type}"
        super().save_policy(path, self.policy)


    # Loads policy from given path
    def load_policy(self, path=None, policy_type=None):
        if(path == None):
            print("[Error] Invalid path - aborting")
            return 

        path = f"{path}policy_{policy_type}"
        
        self.policy = super().load_policy(path, policy_type)


    # Learns a RL policy on provided environment (env_obj)
    def train_rl(
        self, 
        env_obj=None, 
        policy="PPO", 
        policy_type="MlpPolicy",
    ):
        # Record details about policy
        self.policy_details = {
            "env_trained_on": env_obj.env_id,
            "n_envs":env_obj.n_envs,
            "total_timesteps":env_obj.total_timesteps,
            "policy":policy,
            "policy_type":policy_type,
            "logger":env_obj.logger,
            "seed":env_obj.seed,
            "env_kwargs": env_obj.env_kwargs
        }

        self.policy = super().train_rl(env_obj=env_obj, policy=policy, policy_type=policy_type)


    # Plays game with RL policy using OpenAI Gym simulator
    def play_game_with_rl_policy(self, env_obj=None, render_mode="human"):
        super().play_game_with_policy(env_obj=env_obj, render_mode=render_mode, policy=self.policy)