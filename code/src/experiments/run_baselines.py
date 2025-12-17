"""
    Python program for running my KAZ environment baseline tests (i.e. no IRL-based agents) 
    by calling methods in kaz_baselines.py and common.py.
    
    This program:
        - Loads experiment hyperparameters from configuration file
        - Generates X random seeds
        - Runs and repeats the experiment for each seed (Performs all 6 baseline tests per seed)
            - Results are logged to Weights & Biases
        - Stores results as CSV files
        - Calculates the average results 
        - Logs average results to Weights & Biases

"""

import wandb
import time
import torch
import pprint
import os
import sys

# Add parent directory to path so imports work when run from code/src
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

from common.common import *
from baselines.kaz.kaz_baselines import *
import sys 


# Changes the wandb run to allow each test to appear as a new run in W&Bs which allows for easier results comparison at the end
def update_wandb_run(
    use_wandb,
    api_key,
    project_name,
    config,
    experiment_name,
    wandb_run,
    wandb_group,
):
    """ Updates the wandb run's experiment name to ensure different baseline tests have their own wandb run """
    if use_wandb:
        if wandb_run is not None:
            # Need to finish the current run
            wandb_run.finish(
                quiet=True, # to minimise log output
            )

        return setup_wandb(
            api_key=api_key, 
            project_name=project_name, 
            experiment_name=experiment_name, 
            config=config,
            baseline_test=True,
            wandb_group=wandb_group,
        )
    else:
        return None

# Stores mean summary statistics locally
def _store_mean_summary_statistics(
        csv_path,
        baseline_results,
        csv_file_name = "evaluation.csv",
    ):

    """ Stores the averaged baseline results in a CSV file """

    new_csv_path = f"{csv_path}/"
    
    # Create path if it doesn't already exist
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
    
    # split dictionary into labels and values
    labels = baseline_results.keys()
    values = baseline_results.values()

    # CSV Data
    csv_labels = []
    csv_values = []
    for (label, test_result) in zip(labels, values):
        # Setup CSV labels
        csv_labels.append(f"{label}/mean_reward/")
        csv_labels.append(f"{label}/mean_std/")
        csv_labels.append(f"{label}/mean_ep_len/")

        # Setup CSV data
        csv_values.append(str(test_result[0])) # mean reward
        csv_values.append(str(test_result[1])) # mean std
        csv_values.append(str(test_result[2])) # mean ep_len

    # Write to CSV file
    with open(f'{new_csv_path}{csv_file_name}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_labels)
        writer.writerow(csv_values)

    print(f"\n[SUCCESS] Mean Baseline Summary Statistics Saved")


# Runs kaz baseline tests using the user-provided seed. 
# The training/evaluation results are logged to CSV files, and the summary statistics are returned in a dictionary
def run_baselines(
        total_timesteps=100,
        n_envs=4,
        n_eval_episodes=10, 
        max_episode_steps_for_env=None, 
        use_wandb=False,
        user_seed = None,
        project_root = "./baselines/kaz",
        csv_root = "monitor_dir/",
        model_root = "baseline_models/",
        project_name = "kaz_baselines_tracking_vanilla",
        api_key = os.environ.get("WANDB_API_KEY", ""),
        wandb_group = "",
        policy_type = "PPO",
        fixed_horizon_num_steps = IGNORE_HORIZON,
        horizon_choice='variable',
        environment_skill_level='vanilla',
        max_zombies = 10,
    ):

    # Experiment Tracking
    wandb_run = None

    # Paths
    csv_path = f"{project_root}/{horizon_choice}/seed_{user_seed}/{csv_root}"
    model_path = f"{project_root}/{horizon_choice}/seed_{user_seed}/{model_root}"

    # Hyperparameters
    global_total_timesteps = total_timesteps
    global_n_envs = n_envs
    global_n_eval_episodes = n_eval_episodes
    global_seed = user_seed
    global_policy_type = policy_type
    global_fixed_horizon_num_steps = fixed_horizon_num_steps
    global_max_zombies = max_zombies

    # Store baseline results
    baseline_results = {}

    # Output configuration information
    config = {
        "experiment_start_time": int(time.time()),
        "global_total_timesteps": global_total_timesteps,
        "global_n_envs": global_n_envs,
        "global_max_zombies": global_max_zombies,
        "global_n_eval_episodes": global_n_eval_episodes,
        "max_episode_steps_for_env":max_episode_steps_for_env,
        "project_root": project_root,
        "csv_path": csv_path,
        "model_path": model_path,
        "policy_type": global_policy_type,
        "fixed_horizon_num_steps": global_fixed_horizon_num_steps,
        "horizon_choice": horizon_choice,
        "env_names": [
            "SingleKAZ-v0", 
            "DoubleKAZ-v0", 
            "FullObsDoubleKAZ-v0", 
            "SingleRLPolicyDoubleKAZ-v0", 
            "RandomKAZ-v0"
        ],
        "wandb_group":wandb_group,
        "supported_rl_policies":supported_policy_types,
        "using_gpu": torch.cuda.is_available(),
        "user_seed": global_seed,
        "environment_skill_level": environment_skill_level,
    }

    print(f"\n[Starting]\nBaseline Tests - Seed {global_seed}: \n")
    
    # Output config info
    print("\n[Experiment Config Info]")
    pprint.pprint(config)
    print("\n")
    
    
    # Baseline test 1
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_separate_policies_models",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test one, test_separate_policies\n")
    test_separate_policies_path = f"{model_path}test_separate_policies_models/"

    # Notice that this test doesn't need to PZ original KAZ
    baseline_results["test_separate_policies"] = test_separate_policies(
        total_timesteps=global_total_timesteps,
        n_envs=global_n_envs,
        n_eval_episodes=global_n_eval_episodes,
        model_path=test_separate_policies_path,
        policy_type=global_policy_type,
        csv_path=f'{csv_path}test_separate_policies/',
        wandb_run=wandb_run,
        training_env_id="SingleKAZ-v0", 
        eval_env_id="EvalDoubleKAZ-v0",
        user_seed=global_seed,
        fixed_horizon_num_steps=fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")


    # Baseline test 2
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_random_policies",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test two, test_random_policies\n")
    baseline_results["test_random_policies"] = test_random_policies(
        n_envs=global_n_envs,
        n_eval_episodes=global_n_eval_episodes,
        eval_env_id="RandomKAZ-v0",
        path_to_expert=f"{test_separate_policies_path}_archer_{global_policy_type}",
        policy_type=global_policy_type,
        csv_path=f'{csv_path}test_random_policies/',
        wandb_run=wandb_run,
        user_seed=global_seed,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")


    # Baseline test 3
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_same_random_policy",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test three, test_same_random_policy\n")
    baseline_results["test_same_random_policy"] = test_same_random_policy(
        n_envs=global_n_envs,
        n_eval_episodes=global_n_eval_episodes,
        eval_env_id="RandomKAZ-v0",
        path_to_expert=f"{test_separate_policies_path}_archer_{global_policy_type}",
        policy_type=global_policy_type,
        csv_path=f'{csv_path}test_same_random_policy/',
        wandb_run=wandb_run,
        user_seed=global_seed,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")


    # Baseline test 4
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_train_on_different_envs",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test four, test_train_on_different_envs\n")
    test_train_on_different_envs_path = f"{model_path}test_train_on_different_envs_models/"
    baseline_results["test_train_on_different_envs"] = test_train_on_different_envs(
        single_player_policy_type= global_policy_type,
        double_player_policy_type= global_policy_type,
        model_path=test_train_on_different_envs_path,
        csv_path=f'{csv_path}test_train_on_different_envs/',
        wandb_run=wandb_run,
        n_envs=global_n_envs,
        total_timesteps=global_total_timesteps,
        n_eval_episodes=global_n_eval_episodes,
        single_player_env_id="SingleKAZ-v0",
        double_player_env_id="DoubleKAZ-v0",
        eval_env_id="EvalDoubleKAZ-v0",
        user_seed=global_seed,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")


    # Baseline test 5
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_train_on_different_envs_full_observation",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test five, test_train_on_different_envs_full_observation\n")
    test_train_on_different_envs_full_observation_path = f"{model_path}test_train_on_different_envs_full_observation_models/"
    baseline_results["test_train_on_different_envs_full_observation"] = test_train_on_different_envs_full_observation(
        single_player_policy_type = global_policy_type,
        double_player_policy_type = global_policy_type,
        model_path=test_train_on_different_envs_full_observation_path,
        csv_path=f'{csv_path}test_train_on_different_envs_full_observation/',
        wandb_run=wandb_run,
        n_envs=global_n_envs,
        total_timesteps=global_total_timesteps,
        n_eval_episodes=global_n_eval_episodes,
        single_player_env_id="SingleKAZ-v0",
        double_player_env_id="FullObsDoubleKAZ-v0",
        eval_env_id="EvalFullObsDoubleKAZ-v0",
        user_seed=global_seed,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")
    

    # Baseline test 6
    wandb_run = update_wandb_run(
        use_wandb=use_wandb,
        api_key=api_key,
        project_name=project_name,
        config=config,
        experiment_name="test_single_rl_policy_full_observation",
        wandb_run=wandb_run,
        wandb_group=wandb_group,
    )
    print("\n[Starting] Baseline test six, test_single_rl_policy_full_observation\n")
    test_single_rl_policy_full_observation_path = f"{model_path}test_single_rl_policy_full_observation/"
    baseline_results["test_single_rl_policy_full_observation"] = test_single_rl_policy_full_observation(
        env_id="SingleRLPolicyDoubleKAZ-v0",
        n_envs=global_n_envs,
        policy_type=global_policy_type,
        total_timesteps=global_total_timesteps,
        n_eval_episodes=global_n_eval_episodes,
        model_path=test_single_rl_policy_full_observation_path,
        csv_path=f'{csv_path}test_single_rl_policy_full_observation/',
        wandb_run=wandb_run,
        user_seed=global_seed,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        max_zombies = global_max_zombies,
    )
    print("\n[Complete]\n")


    # Output experiment summary

    print(f"\n[Finished] \n Baseline Summary - Seed {user_seed}: \n")
    pprint.pprint(baseline_results)
    
    # Log final results to W&Bs
    if(use_wandb):

        # Change run
        wandb_run = update_wandb_run(
            use_wandb=use_wandb,
            api_key=api_key,
            project_name=project_name,
            config=config,
            experiment_name="experiment_results",
            wandb_run=wandb_run,
            wandb_group=wandb_group,
        )

        # Save Learned Models as Artifacts
        artifact = wandb.Artifact(f'baseline_models_seed_{global_seed}_horizon_{horizon_choice}', type='models')
        artifact.add_dir(model_path)
        wandb_run.log_artifact(artifact)

        # Log results as bar charts
        test_name = "baseline_test"
        table_list = create_tables_from_results(baseline_results, test_name=test_name)
        
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

    return baseline_results


# Runs the baseline tests with different seeds and calculates the average summary statistics
def main(parsed_config_file):

    # Silences the WandB output
    os.environ["WANDB_SILENT"] = str(parsed_config_file["WANDB"]["SILENT"])

    # CSV and model paths
    project_root = parsed_config_file["BASELINE_EXPERIMENT"]["PROJECT_ROOT"]
    csv_root = parsed_config_file["BASELINE_EXPERIMENT"]["CSV_ROOT"]
    model_root = parsed_config_file["BASELINE_EXPERIMENT"]["MODEL_ROOT"]

    # Experiment tracking
    use_wandb = parsed_config_file["WANDB"]["USE_WANDB"]
    
    if use_wandb:
        print("\n\n[Using W&B experiment tracking]\n")

    api_key = parsed_config_file["WANDB"]["API_KEY"]
    project_name = parsed_config_file["WANDB"]["PROJECT_NAME"]

    # Get random seeds
    number_of_test_runs = parsed_config_file["BASELINE_EXPERIMENT"]["NUMBER_OF_TEST_RUNS"]
    print(f"\n[Number of Test Repeats : {number_of_test_runs}]\n")
    seeds = get_random_seeds(
        number_of_test_runs=number_of_test_runs,
        lower_bound = parsed_config_file["BASELINE_EXPERIMENT"]["SEED"]["SEED_LOWER_BOUND"],
        inclusive_upper_bound= parsed_config_file["BASELINE_EXPERIMENT"]["SEED"]["SEED_INCLUSIVE_UPPER_BOUND"]
    )
   
    # Register environments
    max_episode_steps_for_env = None if parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_MAX_EPISODE_STEPS"] == "None" else parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_MAX_EPISODE_STEPS"]
    fixed_horizon_num_steps, horizon_choice = setup_environments(
        max_episode_steps_for_env,
        experiment_name="BASELINE_EXPERIMENT",
        parsed_config_file=parsed_config_file,
    )

    # Set environment skill level (vanilla, medium_knight_better, hard)
    environment_skill_level = parsed_config_file["BASELINE_EXPERIMENT"]["SKILL_LEVEL"]
    project_name = f"{project_name}_{environment_skill_level}"
    select_skill_level(environment_skill_level)
    global_max_zombies = parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_MAX_ZOMBIES"]
    print(f"\n[Playing with Skill Level - {environment_skill_level}]\n")

    # For each seed, run the baselines
    baseline_results = []
    for seed in seeds:
        # Run baselines
        baseline_results.append(
            run_baselines(
                use_wandb=use_wandb,
                max_episode_steps_for_env=max_episode_steps_for_env,
                n_envs=parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_N_ENVS"],
                total_timesteps = parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_TOTAL_TIMESTEPS"],
                n_eval_episodes = parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_N_EVAL_EPISODES"],
                user_seed=seed,
                project_root = project_root,
                csv_root = csv_root, 
                model_root = model_root,
                project_name = project_name,
                api_key = api_key,
                wandb_group=f"seed_{seed}",
                policy_type=parsed_config_file["BASELINE_EXPERIMENT"]["GLOBAL_POLICY_TYPE"],
                fixed_horizon_num_steps= fixed_horizon_num_steps,
                horizon_choice=horizon_choice,
                environment_skill_level=environment_skill_level,
                max_zombies = global_max_zombies,
            ) # n_envs can affect training because it reduces how many steps are made PER environment!
        )

    # Calculate mean summary statistics
    print("\n[Calculating Averaged Baseline Results]\n")
    average_baseline_results = calculate_mean_summary_statistics(
        baseline_results,
    )
    print("\nAveraged Baseline Results: \n")
    pprint.pprint(average_baseline_results)

    # Save Average Baselines Locally
    seeds_used = '_'.join(str(seed) for seed in seeds)
    _store_mean_summary_statistics(
        csv_path= f"{project_root}/{horizon_choice}/average_results_seeds_{seeds_used}/",
        baseline_results=average_baseline_results,
    )

    # Store average baseline results on W&Bs
    if use_wandb:
        wandb_run = update_wandb_run(
            use_wandb=use_wandb,
            api_key=api_key,
            project_name=project_name,
            config={
                "seeds": seeds,
                "number_of_test_runs": number_of_test_runs,
            },
            experiment_name="experiment_results",
            wandb_run=None,
            wandb_group="averaged_results",
        )

        # Log results as bar charts
        test_name = "baseline_test"
        table_list = create_tables_from_results(average_baseline_results, test_name=test_name)
        
        # Reward and std bar chart
        wandb_run.log({
                "Mean Reward Table" : wandb.plot.bar(table_list["mean_reward"], test_name, "mean_reward", title="Mean Reward"),
                "Mean Standard Deviation Table" : wandb.plot.bar(table_list["mean_std"], test_name, "mean_std", title="Mean Standard Deviation"),
                "Mean Episode Length Table" : wandb.plot.bar(table_list["mean_episode_length"], test_name, "mean_episode_length", title="Mean Episode Length"),
            }
        )


if __name__ == "__main__":
    num_cmd_args = len(sys.argv) 
    
    # Get path to main_experiment configuration file
    if(num_cmd_args != 2):
        yaml_file_path = "./configuration/kaz_baselines_config.yaml"
    else:
        yaml_file_path = sys.argv[1]
    
    # Load in configuration file contents
    parsed_config_file = read_in_configuration_file(yaml_file_path)

    # Run Experiment
    main(parsed_config_file)