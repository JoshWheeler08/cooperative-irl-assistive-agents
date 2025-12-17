"""
    Python program for running my main experiments (i.e. tests investigating the performance impact of IRL-based agents) 
    by calling methods in experiment_core.py. 
    
    This program:
        - Loads experiment hyperparameters from configuration file
        - Generates X random seeds
        - Runs and repeats the experiment for each seed (one experiment may test multiple different Assistant agent configurations [RL + IRL])
            - Results are logged to Weights & Biases
        - Stores results as CSV files
        - Calculates the average results 
        - Logs average results to Weights & Biases

"""


import os
import pprint
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

from experiments.experiment_core import *
from common.common import get_random_seeds, calculate_mean_summary_statistics, read_in_configuration_file, setup_environments, select_skill_level
import csv

def _calculate_mean_summary_statistics_per_experiment(
    experiment_names,
    per_seed_results,
):
    """ Calculates the mean summary statistics for each experiment performed in run_main_experiments """
    
    # Stores the averaged results for each experiment
    all_experiment_results = {}

    # For each experiment run (i.e. PPO_AIRL_1e6)
    for experiment_name in experiment_names:
        results = []
        # For each seed 
        for result in per_seed_results:
            # Extract the mean results for this experiment and add to list
            results.append(result[experiment_name])
        
        # Using list of mean results for experiment under different seeds
        all_experiment_results[experiment_name] = calculate_mean_summary_statistics(results)
    
    return all_experiment_results


def _store_mean_summary_statistics_per_experiment(
        csv_path,
        average_test_results,
        experiment_names,
        csv_file_name = "evaluation.csv",
    ):

    """ Stores the averaged test results for each experiment in a CSV file """

    for experiment in experiment_names:
        new_csv_path = f"{csv_path}/{experiment}/"
        
        # Create path if it doesn't already exist
        os.makedirs(os.path.dirname(new_csv_path), exist_ok=True)
        
        # split dictionary into labels and values
        labels = average_test_results[experiment].keys()
        values = average_test_results[experiment].values()

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

        print(f"\n[SUCCESS] Mean Summary Statistics Saved for Experiment {experiment} ")


def run_main_experiments(
        project_root,
        csv_root,
        model_root,
        experiment_names,
        use_wandb,
        project_name,
        global_total_timesteps,
        global_n_envs,
        global_n_eval_episodes,
        global_max_episode_steps,
        global_seeds,
        api_key,
        irl_arguments,
        fixed_horizon_num_steps,
        horizon_choice,
        max_zombies,
    ):

    """ Runs the main experiment for my research which tries different Assistive agent configurations repeated for 3 random seeds """

    # Set obvious hyperparameters which will be fixed during experiments
    env_output = False
    owner_agent_policy_type = final_assistant_policy_type = assistant_rl_policy_type = "MlpPolicy"

    # To store the results for each seed
    per_seed_results = []
    
    # For each seed
    for seed in global_seeds:

        print(f"\n[Starting Main Experiments with seed : {seed}]")
        # To store the results for each seed
        per_experiment_results = {}

        # Run each experiment
        for experiment_name in experiment_names:
            experiment_details = experiment_name.split("_")
            # print(experiment_details) - [PPO, AIRL, 1e6]
            
            # Set up experiment-specific args
            rl_alg = experiment_details[0]
            irl_alg = experiment_details[1]
            current_irl_args = irl_arguments[irl_alg.upper()]
            csv_path = f"{project_root}/{horizon_choice}/seed_{seed}/{experiment_name}/{csv_root}"
            model_path = f"{project_root}/{horizon_choice}/seed_{seed}/{experiment_name}/{model_root}"
            wandb_run = None
            artifact_name = f"learned_models_{seed}_{experiment_name}_horizon_{horizon_choice}"

            experiment_config_info = {
                "experiment_start_time": int(time.time()),
                "max_episode_steps": global_max_episode_steps,
                "max_zombies": max_zombies,
                "total_timesteps": global_total_timesteps,
                "global_n_envs": global_n_envs,
                "irl_alg":irl_alg,
                "irl_arguments": current_irl_args,
                "env_output":env_output,
                "user_seed":seed,
                "n_eval_episodes": global_n_eval_episodes,
                "owner_agent_policy":rl_alg,
                "owner_agent_policy_type":owner_agent_policy_type,
                "final_assistant_policy":rl_alg,
                "final_assistant_policy_type":final_assistant_policy_type,
                "assistant_rl_policy":rl_alg,
                "assistant_rl_policy_type":assistant_rl_policy_type,
                "csv_root": csv_path,
                "model_root": model_path,
                "fixed_horizon_num_steps": fixed_horizon_num_steps,
                "horizon_choice": horizon_choice,
                "env_names": [
                    "SingleKAZ-v0", 
                    "DoubleKAZ-v0",
                ],
                "supported_rl_algorithms":Assistant().supported_rl_algs,
                "supported_irl_algorithms": Assistant().supported_irl_algs,
                "using_gpu": torch.cuda.is_available(),
                "artifact_name":artifact_name,
            }

            # Output W&Bs config info
            print("\n[Experiment Config Info]")
            pprint.pprint(experiment_config_info)
            print("\n")

            if use_wandb:
                # Set up wandb run - check for Weights and Biases usage (https://docs.wandb.ai/ref/python/init)
                # Initialise WandB session
                wandb_run = setup_wandb(
                    api_key=api_key,
                    project_name=project_name,
                    experiment_name=experiment_name,
                    config=experiment_config_info,
                    baseline_test=False,
                    wandb_group=f"seed_{seed}",
                )

            # Run the experiment
            experiment_results = run_kaz_experiment(
                total_timesteps=global_total_timesteps, 
                n_envs_per_training=global_n_envs,
                irl_alg=irl_alg,
                irl_arguments = current_irl_args,
                env_output=env_output,
                seed=seed,
                n_eval_episodes=global_n_eval_episodes,
                wandb_run=wandb_run,
                experiment_name=experiment_name,
                owner_agent_policy=rl_alg,
                owner_agent_policy_type=owner_agent_policy_type,
                final_assistant_policy=rl_alg,
                final_assistant_policy_type=final_assistant_policy_type,
                assistant_rl_policy=rl_alg,
                assistant_rl_policy_type=assistant_rl_policy_type,
                csv_path=csv_path,
                model_path=model_path,
                artifact_name=artifact_name,
                fixed_horizon_num_steps=fixed_horizon_num_steps,
                max_zombies=max_zombies,
            )

            # Store experiment results
            per_experiment_results[experiment_name] = experiment_results

        print(f"\n[SUCCESS] Experiment with seed - {seed} - has ended successfully")
        
        # Store results for this seed
        per_seed_results.append(per_experiment_results)

    print("\n[SUCCESS] All experiments complete! [SUCCESS] \n")

    # Return experiment results
    return per_seed_results


def main(parsed_config_file):
    """ Initialises the hyperparameters before running the main experiment code"""

    # Run all or just one test flag
    full_test = parsed_config_file["EXPERIMENT"]["FULL_TEST"]

    # CSV and model paths
    project_root = parsed_config_file["EXPERIMENT"]["PROJECT_ROOT"]
    csv_root = parsed_config_file["EXPERIMENT"]["CSV_ROOT"]
    model_root = parsed_config_file["EXPERIMENT"]["MODEL_ROOT"]

    # Set up experiment names
    experiment_names = parsed_config_file["EXPERIMENT"]["COMPLETE_EXPERIMENT_LIST"] if full_test else parsed_config_file["EXPERIMENT"]["TEST_EXPERIMENT_LIST"]
    irl_arguments = parsed_config_file["EXPERIMENT"]["IRL_ARGS"] if full_test else parsed_config_file["EXPERIMENT"]["TEST_IRL_ARGS"]

    # Experiment tracking
    use_wandb = parsed_config_file["WANDB"]["USE_WANDB"]
    
    if use_wandb:
        print("\n\n[Using W&B experiment tracking]\n")

    project_name = parsed_config_file["WANDB"]["PROJECT_NAME"]
    api_key = parsed_config_file["WANDB"]["API_KEY"]

    # Silences the WandB output
    os.environ["WANDB_SILENT"] = str(parsed_config_file["WANDB"]["SILENT"])
    
    # Get random seeds
    number_of_test_runs = parsed_config_file["EXPERIMENT"]["NUMBER_OF_TEST_RUNS"]
    print(f"\n[Number of Test Repeats : {number_of_test_runs}]\n")
    seeds = get_random_seeds(
        number_of_test_runs=number_of_test_runs,
        lower_bound = parsed_config_file["EXPERIMENT"]["SEED"]["SEED_LOWER_BOUND"],
        inclusive_upper_bound= parsed_config_file["EXPERIMENT"]["SEED"]["SEED_INCLUSIVE_UPPER_BOUND"],
    )

    # Set meta-hyperparameters
    global_total_timesteps = parsed_config_file["EXPERIMENT"]["GLOBAL_TOTAL_TIMESTEPS"]
    global_n_envs = parsed_config_file["EXPERIMENT"]["GLOBAL_N_ENVS"]
    global_n_eval_episodes = parsed_config_file["EXPERIMENT"]["GLOBAL_N_EVAL_EPISODES"]
    global_max_episode_steps = None if parsed_config_file["EXPERIMENT"]["GLOBAL_MAX_EPISODE_STEPS"] == "None" else parsed_config_file["EXPERIMENT"]["GLOBAL_MAX_EPISODE_STEPS"]
    global_max_zombies = parsed_config_file["EXPERIMENT"]["GLOBAL_MAX_ZOMBIES"]

    # Set environment skill level (vanilla, medium_knight_better, hard)
    environment_skill_level = parsed_config_file["EXPERIMENT"]["SKILL_LEVEL"]
    project_name = f"{project_name}_{environment_skill_level}"
    select_skill_level(environment_skill_level)
    print(f"\n[Playing with Skill Level - {environment_skill_level}]\n")

    # Register the environments
    fixed_horizon_num_steps, horizon_choice = setup_environments(
        global_max_episode_steps,
        experiment_name="EXPERIMENT",
        parsed_config_file=parsed_config_file,
    )
    
    
    # Run experiments
    per_seed_results = run_main_experiments(
        project_root=project_root,
        csv_root=csv_root,
        model_root=model_root,
        experiment_names=experiment_names,
        use_wandb=use_wandb,
        project_name=project_name,
        global_total_timesteps=global_total_timesteps,
        global_n_envs=global_n_envs,
        global_n_eval_episodes=global_n_eval_episodes,
        global_max_episode_steps=global_max_episode_steps,
        global_seeds=seeds,
        api_key=api_key,
        irl_arguments=irl_arguments,
        fixed_horizon_num_steps = fixed_horizon_num_steps,
        horizon_choice=horizon_choice,
        max_zombies=global_max_zombies,
    )


    # Calculate Averaged Test Results
    print("[Calculating Averaged Test Results (over seeds)]")
    average_test_results = _calculate_mean_summary_statistics_per_experiment(
        experiment_names,
        per_seed_results,
    )

    # Store Averaged Test Results Locally
    seeds_used = '_'.join(str(seed) for seed in seeds)
    _store_mean_summary_statistics_per_experiment(
        csv_path= f"{project_root}/{horizon_choice}/average_results_seeds_{seeds_used}/",
        average_test_results=average_test_results,
        experiment_names=experiment_names,
    )

    # Output Averaged Test Results
    for experiment in experiment_names:
        print(f"\n[Outputting Averaged Test Results for Experiment - {experiment}]\n")
        pprint.pprint(average_test_results[experiment])
        print("")

        # Store Averaged Test Results on W&Bs
        if use_wandb:
            # Create new W&Bs run to store results
            wandb_run = setup_wandb(
                api_key=api_key,
                project_name=project_name,
                config={
                    "seeds": seeds,
                    "number_of_test_runs": number_of_test_runs,
                },
                baseline_test=False,
                experiment_name=f"{experiment}_results",
                wandb_group=f"averaged_results",
            )

            # Log results as bar charts
            test_name = "main_experiment"
            table_list = create_tables_from_results(average_test_results[experiment], test_name=test_name)
            
            # Reward and std bar chart
            wandb_run.log({
                    "Mean Reward Table" : wandb.plot.bar(table_list["mean_reward"], test_name, "mean_reward", title="Mean Reward"),
                    "Mean Standard Deviation Table" : wandb.plot.bar(table_list["mean_std"], test_name, "mean_std", title="Mean Standard Deviation"),
                    "Mean Episode Length Table" : wandb.plot.bar(table_list["mean_episode_length"], test_name, "mean_episode_length", title="Mean Episode Length"),
                }
            )

            wandb_run.finish()


if __name__ == "__main__":
    num_cmd_args = len(sys.argv) 
    
    # Get path to main_experiment configuration file
    if(num_cmd_args != 2):
        yaml_file_path = "./configuration/main_experiment_config.yaml"
    else:
        yaml_file_path = sys.argv[1]
        print(f"\n[Using Configuration file - {yaml_file_path}]")
    
    # Load in configuration file contents
    parsed_config_file = read_in_configuration_file(yaml_file_path)

    # Run Experiment
    main(parsed_config_file)
