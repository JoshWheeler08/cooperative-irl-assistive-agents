"""
    Python file containing the core code for running my experiments, 
    which investigate the performance of IRL-based assistive agents in KAZ (or any suitable) environment.  

"""

import os
import time
import pprint
import wandb
import torch

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Logging
from stable_baselines3.common.logger import configure
import imitation.util.logger as imitation_log_code


# My Agents
from wrappers.kaz_training_wrapper import KAZTrainingWrapper
from agents.helpers.env_config import EnvObject
from agents.assistant import Assistant
from agents.owner import Owner

from common.constants import IGNORE_HORIZON
from common.common import setup_wandb, log_results_to_wandb, create_tables_from_results, get_random_seed, register_environments


def _train_owner_agent(
    n_envs_per_training, 
    total_timesteps,
    owner_agent_policy,
    owner_agent_policy_type,
    owner_env_kwargs,
    single_player_env_id,
    logger,
    seed
):
    """ Method for training the owner agent on an arbitrary environment in single player mode """

    # Create Owner
    owner_agent = Owner()

    # Create EnvObject
    owner_agent_env_obj = EnvObject(
        env_id=single_player_env_id,
        n_envs = n_envs_per_training, 
        total_timesteps=total_timesteps, 
        env_kwargs=owner_env_kwargs,
        logger=logger,
        seed=seed,
    )

    print("[Operation] TRAINING THE OWNER AGENT ")

    owner_agent.train_rl(
        owner_agent_env_obj,
        policy=owner_agent_policy, 
        policy_type=owner_agent_policy_type,
    )

    return owner_agent, owner_agent_env_obj


def _train_assistant_agent(
    # Experiment-wide parameters
    n_envs_per_training,
    total_timesteps,
    owner_agent,
    seed,
    # Environment params
    single_player_env_id,
    double_player_env_id,
    assistant_rl_env_kwargs,
    combiner_env_kwargs,
    owner_agent_env_obj,
    # RL Policy Params
    assistant_rl_policy,
    assistant_rl_policy_type,
    final_assistant_policy,
    final_assistant_policy_type,
    # IRL Params
    irl_alg,
    irl_arguments,
    # Loggers
    assistant_rl_logger,
    internal_model_logger,
    combiner_logger,
):
    """ Method for training the assistant agent on an arbitrary environment in both single-player and double-player modes """


    # Create Assistant Agent
    assistant_agent = Assistant()

    # Create Assistant Agent RL Environment (single player mode)
    assistant_rl_env_obj = EnvObject(
        env_id=single_player_env_id,
        n_envs=n_envs_per_training,
        total_timesteps=total_timesteps,
        env_kwargs=assistant_rl_env_kwargs,
        seed=seed,
        logger=assistant_rl_logger,
    )

    # Create Assistant Agent Final RL Environment (double player mode)
    combiner_env_obj = EnvObject(
        env_id=double_player_env_id,
        n_envs=n_envs_per_training,
        total_timesteps=total_timesteps,
        env_kwargs=combiner_env_kwargs,
        wrapper_class=KAZTrainingWrapper,
        seed=seed,
        logger=combiner_logger,
    )

    # Change logger since this env object was also used for training the Owner agent
    owner_agent_env_obj.logger = internal_model_logger

    print("[Operation] TRAINING THE ASSISTANT AGENT")

    assistant_agent.train(
        assistant_env_obj = assistant_rl_env_obj,
        internal_model_env_obj = owner_agent_env_obj,
        combiner_env_obj = combiner_env_obj,
        expert_owner_agent=owner_agent,
        assistant_rl_policy=assistant_rl_policy,
        assistant_rl_policy_type=assistant_rl_policy_type,
        combiner_rl_policy=final_assistant_policy,
        combiner_rl_policy_type=final_assistant_policy_type,
        irl_alg=irl_alg,
        irl_arguments=irl_arguments,
    ) # Owner agent could be replaced with real-life trajectories - as an extension - to make my results more reliable

    return assistant_agent, assistant_rl_env_obj, combiner_env_obj


def _evaluate_agents(
    # Owner parameters
    owner_agent, 
    owner_agent_env_obj,
    # Assistant parameters
    assistant_agent,
    assistant_rl_env_obj,
    combiner_env_obj,
    combiner_eval_env_id,
    # Experiment-wide parameters
    n_eval_episodes,
    csv_path,
):

    """ Method for evaluating the Owner and Assistant agents performance in the selected environment using SB3's evaluate_policy method """

    print(f"[Evaluation] Evaluating Trained Agents - n_eval_episodes = {n_eval_episodes}")

    results = {}

    eval_paths = {
        "owner_agent": f"{csv_path}evaluation/final/knight/",
        "assistant_agent_final_policy": f"{csv_path}evaluation/final/archer/",
        "assistant_agent_rl": f"{csv_path}evaluation/rl/archer/",
        "assistant_agent_internal_model": f"{csv_path}evaluation/irl/archer/",
    }

    # Evaluate Owner RL policy
    owner_agent_env_obj.logger = None # Don't need a logger for this
    results["owner_agent"] = owner_agent.evaluate_policy(
        env_obj=owner_agent_env_obj, 
        n_eval_episodes=n_eval_episodes, 
        csv_path= eval_paths["owner_agent"],
    )

    # Just for testing - owner_agent.play_game_with_rl_policy(env_obj = owner_agent_env_obj)

    # Evaluate Assistant RL Policy
    assistant_rl_env_obj.logger = None # Don't need a logger for this
    results["assistant_agent_rl"]  = assistant_agent.evaluate_rl_policy(
        env_obj=assistant_rl_env_obj, 
        n_eval_episodes=n_eval_episodes, 
        csv_path=eval_paths["assistant_agent_rl"]
    )

    # Evaluate Assistant Internal Model
    results["assistant_agent_internal_model"] = assistant_agent.evaluate_internal_model_policy(
        env_obj=owner_agent_env_obj, 
        n_eval_episodes=n_eval_episodes, 
        csv_path=eval_paths["assistant_agent_internal_model"]
    )


    # Evaluate Final Assistant Policy
    combiner_env_obj.logger = None
    combiner_env_obj.env_id = combiner_eval_env_id # step function returns joint reward rather than just assistant's reward per timestep
    results["assistant_agent_final_policy"] = assistant_agent.evaluate_policy(
        env_obj=combiner_env_obj, 
        n_eval_episodes=n_eval_episodes, 
        csv_path=eval_paths["assistant_agent_final_policy"]
    )

    print("[Finished Evaluation]")

    return results, eval_paths


def _wandb_integration(
    wandb_run,
    model_path,
    owner_agent_log_folder,
    assistant_rl_log_folder,
    internal_model_log_folder,
    combiner_log_folder,
    evaluation_results,
    eval_paths,
    artifact_name,
):

    """ Method for enabling W&Bs experiment tracking which will log and plot the provided metric data for easy analysis """

    # Save Learned Models as Artifacts
    artifact = wandb.Artifact(artifact_name, type='models')
    artifact.add_dir(model_path)
    wandb_run.log_artifact(artifact)

    # Log performance metrics stored in CSV files
    default_evaluation_csv_name = "evaluation.csv"

    log_results_to_wandb({
            # Log owner training
            f"owner":f"{owner_agent_log_folder}/progress.csv",
            
            # Log assistant training
            f"assistant/rl":f"{assistant_rl_log_folder}/progress.csv",
            f"assistant/irl":f"{internal_model_log_folder}/progress.csv",
            f"assistant":f"{combiner_log_folder}/progress.csv",

            # Log evaluations
            f"evaluation/owner":f"{eval_paths['owner_agent']}{default_evaluation_csv_name}",
            f"evaluation/assistant/rl":f"{eval_paths['assistant_agent_rl']}{default_evaluation_csv_name}",
            f"evaluation/assistant/irl":f"{eval_paths['assistant_agent_internal_model']}{default_evaluation_csv_name}",
            f"evaluation/assistant":f"{eval_paths['assistant_agent_final_policy']}{default_evaluation_csv_name}",
        },
        wandb_run=wandb_run,
    )

    # Log results as bar charts
    test_name = "main_experiment"
    table_list = create_tables_from_results(evaluation_results, test_name=test_name)
    
    # Reward and std bar chart
    wandb_run.log({
            "Mean Reward Table" : wandb.plot.bar(table_list["mean_reward"], test_name, "mean_reward", title="Mean Reward"),
            "Mean Standard Deviation Table" : wandb.plot.bar(table_list["mean_std"], test_name, "mean_std", title="Mean Standard Deviation"),
            "Mean Episode Length Table" : wandb.plot.bar(table_list["mean_episode_length"], test_name, "mean_episode_length", title="Mean Episode Length"),
        }
    )

    wandb_run.finish(
        quiet=True,
    )


def run_kaz_experiment(
        # Experiment-wide parameters
        total_timesteps=100,
        n_envs_per_training=4,
        irl_alg="BC",
        irl_arguments = None,
        env_output=True,
        seed=42,
        n_eval_episodes=10,
        wandb_run=None,
        experiment_name="test",
        fixed_horizon_num_steps = IGNORE_HORIZON,
        max_zombies = 10,
        # Model parameters
        owner_agent_policy = "PPO",
        owner_agent_policy_type = "MlpPolicy",
        final_assistant_policy = "PPO",
        final_assistant_policy_type = "MlpPolicy",
        assistant_rl_policy = "A2C",
        assistant_rl_policy_type = "MlpPolicy",
        # Paths
        csv_path="./main_experiment_output/monitor_dirs/",
        model_path = "./main_experiment_output/learned_models/",
        artifact_name = "learned_models",
    ):

    """ Main method for running the KAZ experiment """

    print(f"[Starting Experiment] - {experiment_name} - with seed - {seed}")

    # Set up Owner Agent environment
    knight_env_kwargs = {
        "type_of_player":"knight",
        "env_output":env_output,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    # Create Logger for Owner Agent
    owner_agent_log_folder = f"{csv_path}owner"
    owner_logger = configure(owner_agent_log_folder, ["csv", "stdout"])

    # Train Owner agent
    owner_agent, owner_agent_env_obj = _train_owner_agent(
        n_envs_per_training=n_envs_per_training,
        total_timesteps=total_timesteps,
        owner_agent_policy=owner_agent_policy,
        owner_agent_policy_type=owner_agent_policy_type,
        owner_env_kwargs=knight_env_kwargs,
        single_player_env_id="SingleKAZ-v0",
        logger=owner_logger,
        seed=seed,
    )
    

    # Creating Archer RL environment (Single Player)
    archer_env_kwargs = {
        "type_of_player": "archer",
        "env_output": env_output,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }


    # Creating Archer + Knight Environment (Double Player)
    combiner_env_kwargs = {
        "knight_policy": owner_agent.policy, # the FIXED owner policy will control the knight during the Assistant's training
        "env_output":env_output,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }


    # Create Loggers for Assistant Agent
    print("[Operation] Initialising Assistant Loggers")
    assistant_rl_log_folder = f"{csv_path}_assistant_rl"
    assistant_rl_logger = configure(assistant_rl_log_folder, ["csv", "stdout"])

    internal_model_log_folder = f"{csv_path}_internal_model"
    internal_model_logger = imitation_log_code.configure(internal_model_log_folder, ["csv", "stdout"]) 
    # note that I must use imitation.util.logger.configure instead of SB3's implementation to ensure the logger is accumulate_means()-compatible
    # (https://imitation.readthedocs.io/en/latest/_api/imitation.util.logger.html#imitation.util.logger.HierarchicalLogger)

    combiner_log_folder = f"{csv_path}assistant"
    combiner_logger = configure(combiner_log_folder, ["csv", "stdout"])

    # Train Assistant Agent
    assistant_agent, assistant_rl_env_obj, combiner_env_obj = _train_assistant_agent(
            single_player_env_id="SingleKAZ-v0",
            double_player_env_id="DoubleKAZ-v0",
            n_envs_per_training=n_envs_per_training,
            total_timesteps=total_timesteps,
            owner_agent=owner_agent,
            assistant_rl_policy=assistant_rl_policy,
            assistant_rl_policy_type=assistant_rl_policy_type,
            final_assistant_policy=final_assistant_policy,
            final_assistant_policy_type=final_assistant_policy_type,
            irl_alg=irl_alg,
            irl_arguments=irl_arguments,
            assistant_rl_env_kwargs = archer_env_kwargs,
            combiner_env_kwargs=combiner_env_kwargs,
            owner_agent_env_obj=owner_agent_env_obj,
            seed = seed,
            assistant_rl_logger=assistant_rl_logger,
            internal_model_logger=internal_model_logger,
            combiner_logger=combiner_logger,
    )

    # Evaluate Agents Policies
    evaluation_results, eval_paths = _evaluate_agents(
        owner_agent=owner_agent,
        owner_agent_env_obj=owner_agent_env_obj,
        assistant_agent=assistant_agent,
        assistant_rl_env_obj=assistant_rl_env_obj,
        combiner_env_obj=combiner_env_obj,
        n_eval_episodes=n_eval_episodes,
        csv_path=csv_path,
        combiner_eval_env_id="EvalDoubleKAZ-v0",
    )

    # Save Agent Policies for Reproducibility
    print("[Saving Policies]")
    
    owner_save_path = f"{model_path}owner_agent/"
    assistant_save_path = f"{model_path}assistant_agent/"

    owner_agent.save_policy(
        path=owner_save_path,
        policy_type=owner_agent_policy,
    )

    assistant_agent.save_policy(
        path=assistant_save_path,
        rl_policy_type=assistant_rl_policy,
        final_policy_type=final_assistant_policy,
    )

    
    # Output Evaluation Results
    print("\n[Experiment Complete]")
    print("Results Summary: \n")
    pprint.pprint(evaluation_results)
    #print(f"\n User seed = {seed} \n")

    
    # Log results to W&Bs
    if wandb_run is not None:
        _wandb_integration(
            wandb_run,
            model_path,
            owner_agent_log_folder,
            assistant_rl_log_folder,
            internal_model_log_folder,
            combiner_log_folder,
            evaluation_results,
            eval_paths,
            artifact_name,
        )

    return evaluation_results

# Example test run of the main experiment
def main():
    """ Test run """

    # Set up hyperparameters:
    # Experiment Tracking
    use_wandb = False
    wandb_run = None
    project_name = "kaz_main_experiment"
    experiment_name = "test_run"
    wandb_group = "main_experiment"

    # Experiment-wide parameters
    max_episode_steps=None
    total_timesteps=10000
    n_envs_per_training=4
    irl_alg="BC"
    env_output=False
    user_seed = get_random_seed(lower_bound=0, inclusive_upper_bound=500)
    n_eval_episodes=100
    artifact_name = "learned_models"
    
    # Model parameters
    owner_agent_policy = "PPO"
    owner_agent_policy_type = "MlpPolicy"
    final_assistant_policy = "PPO"
    final_assistant_policy_type = "MlpPolicy"
    assistant_rl_policy = "PPO"
    assistant_rl_policy_type = "MlpPolicy"

    # Paths
    csv_path="./test_experiment_core_output/monitor_dirs/"
    model_path = "./test_experiment_core_output/learned_models/"

    # Check for Weights and Biases usage (https://wandb.ai/site)
    if use_wandb:
        # Initialise WandB session
        wandb_run = setup_wandb(
            api_key=os.environ.get("WANDB_API_KEY", ""),
            project_name=project_name,
            experiment_name=experiment_name,
            config={
                "experiment_start_time": int(time.time()),
                "max_episode_steps": max_episode_steps,
                "total_timesteps": total_timesteps,
                "global_n_envs": n_envs_per_training,
                "irl_alg":irl_alg,
                "env_output":env_output,
                "user_seed":user_seed,
                "n_eval_episodes": n_eval_episodes,
                "owner_agent_policy":owner_agent_policy,
                "owner_agent_policy_type":owner_agent_policy_type,
                "final_assistant_policy":final_assistant_policy,
                "final_assistant_policy_type":final_assistant_policy_type,
                "assistant_rl_policy":assistant_rl_policy,
                "assistant_rl_policy_type":assistant_rl_policy_type,
                "csv_root": csv_path,
                "model_root": model_path,
                "env_names": [
                    "SingleKAZ-v0", 
                    "DoubleKAZ-v0",
                ],
                "supported_rl_policies":Owner().supported_rl_algs,
                "using_gpu": torch.cuda.is_available(),
            },
            baseline_test=False,
            wandb_group=wandb_group
        )

    # Register the environments!
    register_environments(max_episode_steps=max_episode_steps)

    # Run the experiment
    run_kaz_experiment(
        total_timesteps=total_timesteps, 
        n_envs_per_training=n_envs_per_training,
        irl_alg=irl_alg,
        env_output=env_output,
        seed=user_seed,
        n_eval_episodes=n_eval_episodes,
        wandb_run=wandb_run,
        experiment_name=experiment_name,
        owner_agent_policy=owner_agent_policy,
        owner_agent_policy_type=owner_agent_policy_type,
        final_assistant_policy=final_assistant_policy,
        final_assistant_policy_type=final_assistant_policy_type,
        assistant_rl_policy=assistant_rl_policy,
        assistant_rl_policy_type=assistant_rl_policy_type,
        csv_path=csv_path,
        model_path=model_path,
        artifact_name=artifact_name,
    )


if __name__ == "__main__":
    main()    