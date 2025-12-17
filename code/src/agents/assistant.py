import gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from irl_training.behavioral_cloning import MyBC
from irl_training.density_reward import MyDensity
from irl_training.gail import MyGAIL
from irl_training.airl import MyAIRL
from irl_training.dagger import MyDAgger
from irl_training.preference_comparisons import MyPrefComp

from agents.agent_base import Agent
from imitation.algorithms.bc import reconstruct_policy


class Assistant(Agent):

    """ Python Class for representing the IRL-based assistive system  """

    def __init__(self, name="Assistant_Agent_1"):
        super().__init__(name=name)

        self.rl_policy = None # Assistant's learned policy for acting in the environment 
        self.internal_model_policy = None # Assistant's approximated model of the Owner agent
        
        self.supported_irl_algs = {     # Supported IRL algorithms
            "BC": MyBC, 
            "Density": MyDensity,
            "GAIL": MyGAIL,
            "AIRL" : MyAIRL,
            "DAgger": MyDAgger,
            "PrefComp" : MyPrefComp,
        }


    # Access Properties

    def rl_policy(self):
        return self.rl_policy

    def internal_model_policy(self):
        return self.internal_model_policy

    def training_details(self):
        return [self.rl_policy_details, self.internal_model_policy_details, self.policy_details]


    # Learn Assistant's RL Policy for acting in the environment
    def train_rl(self, env_obj, policy="PPO", policy_type="MlpPolicy"):
        
        self.rl_policy_details = {
            "env_trained_on": env_obj.env_id,
            "n_envs":env_obj.n_envs,
            "total_timesteps":env_obj.total_timesteps,
            "policy": policy,
            "policy_type":policy_type,
            "logger":env_obj.logger,
            "seed":env_obj.seed,
            "env_kwargs": env_obj.env_kwargs 
        }

        return super().train_rl(
            env_obj, 
            policy, 
            policy_type, 
        )

    # Learn Assistant's internal model of the Owner agent
    def _train_internal_model(self, env_obj, irl_alg, irl_arguments, expert_owner_agent):
        if(expert_owner_agent == None): 
            print("[Error] Invalid Owner Agent passed to train_internal_model() - aborting")
            return None

        if env_obj == None:
            print("[Error] Invalid environment object passed in")
            return None

        if irl_alg in self.supported_irl_algs:
            
            irl_function = self.supported_irl_algs[irl_alg] # which IRL algorithm to use

            # Storing details about the learned model
            self.internal_model_policy_details = {
                "env_trained_on": env_obj.env_id,
                "n_envs":env_obj.n_envs,
                "irl_alg_used":irl_alg,
                "irl_arguments": irl_arguments,
                "expert_owner_agent_identifier":expert_owner_agent.name,
                "logger": env_obj.logger,
                "seed": env_obj.seed,
                "env_kwargs": env_obj.env_kwargs,
            }

            try:
                irl_class = irl_function( # Initialise IRL algorithm 
                    owner_agent = expert_owner_agent, 
                    env_obj=env_obj,
                    training_args=irl_arguments,
                )
                policy_object = irl_class.train()   # Run IRL algorithm 

                if(irl_alg == "BC"):
                    self.internal_model_policy_object = policy_object
                    return policy_object.policy
                else:
                    return policy_object

            except Exception as e:
                print(f"[Error] Problem using IRL algorithm (error message: {e})")
                return None
        else:
            print(f"[Error] IRL algorithm not supported - {irl_alg}")
            return None


    # Learn Assistant's final policy on environment by combining observations 
    # with the output of its rl_policy and internal_model (see KAZTrainingWrapper)
    def _combine_policies(self, env_obj, policy, policy_type):

            # Record details about policy
            self.policy_details = {
                "env_trained_on": env_obj.env_id,
                "n_envs":env_obj.n_envs,
                "total_timesteps":env_obj.total_timesteps,
                "policy":policy,
                "policy_type":policy_type,
                "env_kwargs": env_obj.env_kwargs,
                "wrapper_class": env_obj.wrapper_class,
                "wrapper_kwargs":env_obj.wrapper_kwargs,
                "seed": env_obj.seed,
                "logger": env_obj.logger,
            }

            return super().train_rl(
                env_obj,
                policy,
                policy_type,
            )


    # Learns all three of the Assistant agent's policies (RL, IRL, RL+IRL)
    def train(self,
            assistant_env_obj=None,
            internal_model_env_obj=None,
            combiner_env_obj=None,
            expert_owner_agent=None,
            assistant_rl_policy="A2C",
            assistant_rl_policy_type="MlpPolicy",
            combiner_rl_policy="PPO",
            combiner_rl_policy_type="MlpPolicy", 
            irl_alg = "BC",
            irl_arguments=None,
    ):

        if assistant_env_obj == None or internal_model_env_obj == None or combiner_env_obj == None:
            print("Issue with one of the env_objects")
            return

        # Use RL algorithm to learn Assistant's RL policy for acting in the environment 
        print("[Operation] Training Assistant RL Agent")

        self.rl_policy = self.train_rl(
            assistant_env_obj,
            assistant_rl_policy, 
            assistant_rl_policy_type,
        )

        if self.rl_policy != None:

            print("[SUCCESS] Assistant RL agent trained")
            
            if expert_owner_agent == None:
                print("[ERROR] train() needs a valid owner agent in order to train the assistant's internal model - aborting")
                return

            print("[Operation] Training Assistant internal model")
            # Learn approximation for Owner agent's policy based on expert demonstrations
            self.internal_model_policy = self._train_internal_model(
                internal_model_env_obj,
                irl_alg,
                irl_arguments,
                expert_owner_agent, 
            )

            if self.internal_model_policy != None:
                print("[SUCCESS] Assistant internal model trained")

                # Using an OpenAI Gym wrapper to augment the observation returned to the agent by the environment:
                # [Observation, RL policy(Archer_Observation), IRL-learned policy(Knight_Observation)] 
                wrapper_kwargs = {
                    "rl_policy":self.rl_policy,
                    "internal_model_policy":self.internal_model_policy
                }

                combiner_env_obj.wrapper_kwargs = wrapper_kwargs

                # Learning Assistant's final policy for determining its actions in the game environment 
                # when playing with the Owner agent
                print("[Operation] Training Assistant Agent Final Policy")
                self.policy = self._combine_policies(
                    combiner_env_obj,
                    combiner_rl_policy, 
                    combiner_rl_policy_type,
                )

                if self.policy != None:
                    print("[SUCCESS] Assistant policy created")
                else:
                    print("[Error] Issue training Assistant agent when combining policies")
            
            else:
                print("[Error] Issue training internal model of Assistant agent")
        else:
            print("[Error] Issue training RL policy for Assistant agent")
            


    # Evaluate Polices

    def evaluate_policy(self, env_obj, n_eval_episodes=10, csv_path="./"):
        return super().evaluate_policy(env_obj, n_eval_episodes, self.policy, csv_path)
    
    def evaluate_rl_policy(self, env_obj, n_eval_episodes=10, csv_path="./"):
        return super().evaluate_policy(env_obj, n_eval_episodes, self.rl_policy, csv_path)

    def evaluate_internal_model_policy(self, env_obj, n_eval_episodes=10, csv_path="./"):
            return super().evaluate_policy(env_obj, n_eval_episodes, self.internal_model_policy, csv_path)



    # Save Policies

    def save_policy(self, path=None, rl_policy_type="", final_policy_type=""):
        
        if(path == None):
            print("[Error] Invalid Path - aborting save_policy()")
            return

        if(self.policy != None and self.rl_policy != None and self.internal_model_policy != None): 
            # Create directory
            try:
                # Save policies 
                super().save_policy(f"{path}final_policy_{final_policy_type}", self.policy)

                self.save_rl_policy(path, rl_policy_type)

                self.save_internal_model_policy(path)

                print("Save operation finished")

            except Exception as e:
                print(f"[Error] Failed to create directory (error message : {e})")
        else:
            print("[Error] Not all policies in the assistant have been trained yet!")


    def save_rl_policy(self, path=None, rl_policy_type=""):
        if(path == None):
            print("[Error] Invalid path - aborting")
            return 

        path = f"{path}rl_policy_{rl_policy_type}"
        
        super().save_policy(path, self.rl_policy)


    def save_internal_model_policy(self, path=None):
        if(path == None):
            print(f"[Error] Cannot save internal policy due to missing path")

        irl_used = self.internal_model_policy_details["irl_alg_used"]

        if(irl_used in self.supported_irl_algs):
            path = f"{path}internal_model_policy_{irl_used}"

            if(irl_used == "BC"):
                if(self.internal_model_policy_object == None):
                    print(f"[Error] Cannot save internal policy due to missing internal model policy object")
                else:
                    self.internal_model_policy_object.save_policy(path) # Have to use imitation's own API
                    print(f"Success, policy details saved at {path}")
            else:
                super().save_policy(path, self.internal_model_policy) # its just another SB3-trained agent
                
        else:
            print("[Error] irl_alg not recognised in save_internal_model_policy - aborting operation")
        


    # Load Policies

    def load_policy(self, path=None, final_policy_type="PPO", rl_policy_type="PPO", irl_alg=None, irl_policy_type=None):
        
        if(path == None):
            print("[Error] Invalid path - aborting load_policy()")
            return None
        
        # Need to load all components - assuming path is to directory
        self.policy = super().load_policy(f"{path}final_policy_{final_policy_type}", final_policy_type)
        
        self.load_rl_policy(path, rl_policy_type)
        
        self.load_internal_model_policy(path, irl_alg, irl_policy_type)

        if(self.internal_model_policy == None or self.rl_policy == None or self.policy == None):
            print("[Error] Unable to load all policies")
        else:
            print("Success, all policies loaded")
        

    def load_rl_policy(self, path=None, policy_type=None):
        if(path == None):
            print("[Error] Invalid path - aborting")
            return 

        path = f"{path}rl_policy_{policy_type}"

        self.rl_policy = super().load_policy(path, policy_type)


    def load_internal_model_policy(self, path=None, irl_alg=None, policy_type=None):
        if(path == None):
            print(f"[Error] Cannot load internal policy due to missing path")
            return

        if(irl_alg == None or irl_alg not in self.supported_irl_algs):
            print(f"[Error] Cannot load internal policy due to invalid irl_arg")
            return

        path = f"{path}internal_model_policy_{irl_alg}"

        if(irl_alg == "BC"):
            self.internal_model_policy = reconstruct_policy(path)
        else:
            self.internal_model_policy = ActorCriticPolicy.load(path)

        print(f"Success, policy details loaded from {path}")
            


    # Play Games with Policies

    def play_game_with_rl_policy(self, env_obj, render_mode="human"):
        super().play_game_with_policy(env_obj, render_mode, self.rl_policy)

    # Custom play game method
    def play_game_with_final_policy(self, env_obj=None, render_mode="human"):
        
        # Needs to be a double-player environment because the Assistant's policy relies on observations 
        # from both the Assistant and Owner agent
        if(env_obj.wrapper_kwargs == None):
            env_obj.wrapper_kwargs = {
                "rl_policy": self.rl_policy,
                "internal_model_policy":self.internal_model_policy,
                "running_in_gym_flag":True,
            }

        # Check for wrapper class
        if(env_obj.wrapper_class == None):
            print("Invalid Wrapper Class")
            return
        
        try:
            # Make env with Wrapper
            env = gym.make(id=env_obj.env_id, **env_obj.env_kwargs, render_mode=render_mode, running_in_gym_flag=True)
            env = env_obj.wrapper_class(env, **env_obj.wrapper_kwargs)

            # Play game
            total_timesteps = env_obj.total_timesteps

            obs, info = env.reset()
            for i in range(total_timesteps):
                action, _states = self.policy.predict(obs) # get action 

                obs, rewards, done, info = env.step(action) # see how environment reacts when action taken

                if(done):
                    obs, info = env.reset()

            env.close()
        except Exception as e:
            print(f"[Error] Issue creating environment for play_game_with_final_policy() (error message : {e})")


    def play_game_with_internal_model_policy(self, env_obj=None, render_mode="human"):
        super().play_game_with_policy(env_obj, render_mode, self.internal_model_policy)
