"""
    This is a Python file for storing utility functions commonly used in my experiments.
"""

import wandb
import numpy as np
import csv
import os
import random
import yaml 
import shutil

from common.constants import IGNORE_HORIZON
from gym.envs.registration import register


def register_environments(
    max_episode_steps=None,
    env_entry_points=None,
):
    """ Register Environment for training Owner and Assistant RL Policy """
    # max_episode_steps = None means gym.make won't put a timelimit wrapper around the environment, 
    # which can end the game before it has actually finished!

    for env_id, env_entry_point in env_entry_points.items():
        register(
            id=env_id,
            entry_point=env_entry_point,
            max_episode_steps=max_episode_steps,
        )


def setup_environments(max_episode_steps_for_env, experiment_name, parsed_config_file):
    """ Initialises the game environments required for this experiment using the information stored in a parsed configuration file """

    # Deciding whether to use a fixed or variable-horizon KAZ
    horizon_choice = parsed_config_file[experiment_name]["HORIZON"]["OPTION"]

    if horizon_choice == 'fixed':
        fixed_horizon_num_steps = parsed_config_file[experiment_name]["HORIZON"]["NUM_STEPS"]
        env_entry_points = parsed_config_file[experiment_name]["HORIZON"]["HORIZON_PATHS"]["FIXED_HORIZON_PATHS"] # controls which version of KAZ to use
    else:
        horizon_choice = 'variable'
        fixed_horizon_num_steps = IGNORE_HORIZON
        env_entry_points = parsed_config_file[experiment_name]["HORIZON"]["HORIZON_PATHS"]["VARIABLE_HORIZON_PATHS"]
    
    register_environments(
        max_episode_steps=max_episode_steps_for_env,
        env_entry_points = env_entry_points,
    )

    print(f"\n[Using {horizon_choice.upper()} horizon environments]\n")

    return fixed_horizon_num_steps, horizon_choice


def log_results_to_wandb(log_details_dict, wandb_run):
    """ Logs the requested metrics onto the W&B's dashboard """
    for experiment_name, path in log_details_dict.items():
        _log_csv_file_to_wandb(path, experiment_name, wandb_run)


def _log_csv_file_to_wandb(csv_path, experiment_name, wandb_run):
    "Takes CSV file and converts into format compatible with W&Bs"
    with open(csv_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        headers_seen = False
        for row in datareader:
            if(not(headers_seen)): 
                # Extracting headers and prepending experiment name
                headers = [f'{experiment_name}/charts/{header}' for header in row]
                headers_seen = True
            else:
                # Replaces empty strings with 0
                remove_nulls = lambda str : 0 if str == '' else str 
                filtered_row = map(remove_nulls, row)
                
                # Converting strings to numeric values
                values = [float(value) for value in filtered_row]
                
                # Extracting values and logging to W&Bs
                wandb_run.log(dict(zip(headers, values)))

    print(f"\n{csv_path} successfully logged under experiment name: {experiment_name}")


def setup_wandb(api_key, project_name, experiment_name, config, baseline_test=True, wandb_group="None"):
    """ Sets up W&B to allow experiment tracking"""
    # API key
    os.environ["WANDB_API_KEY"] = api_key
    
    if baseline_test:
        tags = ["baseline"]
    else:
        tags = ["new_algorithm"]

    return wandb.init(
        project=project_name,
        name=experiment_name,
        config=config,
        tags=tags,
        group=wandb_group,
    )


def create_tables_from_results(test_results, test_name="baseline_test"):
    """ Reformats the test_results to be used to create wandb tables """

    # split dictionary into labels and values
    labels = test_results.keys()
    results = test_results.values()

    mean_rewards = []
    mean_stds = []
    mean_ep_lens = []

    # Extract data points
    for test_result in results:
        mean_rewards.append(test_result[0])
        mean_stds.append(test_result[1])
        mean_ep_lens.append(test_result[2])

    # mean_rewards, mean_stds, mean_ep_lens = [(reward, std, ep_len) for [reward, std, ep_len] in values]

    # Iterate through creating the tables
    tables_to_be_created = [mean_rewards, mean_stds, mean_ep_lens]
    column_names = ["mean_reward", "mean_std", "mean_episode_length"]
    table_list = {}

    for (data, column_name) in zip(tables_to_be_created, column_names):
        labels_with_data = [[label, d] for (label, d) in zip(labels, data)]
    
        table_list[column_name] = wandb.Table(
            data=labels_with_data, 
            columns = [test_name, column_name]
        )
        
    return table_list


def get_random_seed(lower_bound=0, inclusive_upper_bound=500):
    """ Returns a random seed for use in the environment spawning process and for initialising the SB3 agents (PPO/DQN)"""
    return random.randint(lower_bound, inclusive_upper_bound)


def calculate_mean_summary_statistics(baseline_results):
    """ Takes all of the baseline results measured for different seeds and produces the average baseline result """
    number_of_test_runs = len(baseline_results)

    if(number_of_test_runs <= 1):
        return baseline_results[0]
    
    averaged_results = baseline_results[0]

    # Sum up all the corresponding experiment values to get their totals
    for project_run in baseline_results[1:]: # skipping first list because we have assigned this to average_results
        for experiment in project_run:
            new_values = project_run[experiment]
            running_total_values = averaged_results[experiment]
            new_running_total_values = [sum(x) for x in zip(new_values, running_total_values)] # element-wise sum
            averaged_results[experiment] = new_running_total_values

    # Calculate the average
    for experiment in averaged_results:
        values = averaged_results[experiment]
        averaged_results[experiment] = [value / number_of_test_runs  for value in values]
        
    return averaged_results


def get_random_seeds(number_of_test_runs, lower_bound=0, inclusive_upper_bound=500):
    """ Returns a list, of length 'number_of_test_runs', containing no duplicate random seeds """
    seeds = []
    # Get the seeds (no duplicates allowed)
    while(len(seeds) != number_of_test_runs):
        new_seed = get_random_seed(
            lower_bound=lower_bound,
            inclusive_upper_bound=inclusive_upper_bound,
        )

        if new_seed not in seeds:
            seeds.append(new_seed)
    return seeds


def read_in_configuration_file(yaml_file_path):
    """ Reads in YAML configuration file which will contain hyperparameter values """

    with open(yaml_file_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"[Error] Issue loading yaml configuration file at path - {yaml_file_path} ({exc})")
            return None


def select_skill_level(skill_level):
    """ Function for selecting the skill level to be used by the game environment """
    
    path_to_constants_file = f"./environments/kaz/core/src/constants.py"

    # Remove existing constants file
    if os.path.exists(path_to_constants_file):
        os.remove(path_to_constants_file)
    
    # Copy and rename new constants file
    path_to_skill_level_details = f"./environments/kaz/core/src/skill_levels/{skill_level}.py"
    shutil.copy(path_to_skill_level_details, path_to_constants_file)


def process_evaluation_results(episode_rewards, episode_lengths, csv_path):
    """ Processes the evaluation results produced by stable_baselines3.common.evaluation.evaluate_policy, and logs to a CSV file"""

    # Calculate summary statistics
    mean_reward = np.mean(episode_rewards)
    mean_episode_length = np.mean(episode_lengths)
    std_reward = np.std(episode_rewards)


    # Log episode rewards/lengths to CSV file
    # Insert CSV headers
    episode_rewards.insert(0, 'evaluation/ep_reward')
    episode_lengths.insert(0, 'evaluation/ep_length')
    
    csv_file_name = "evaluation.csv"

    # Create path if it doesn't already exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # Write to CSV file
    with open(f'{csv_path}{csv_file_name}', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(episode_rewards, episode_lengths))

    # print(episode_rewards)

    return mean_reward, std_reward, mean_episode_length, csv_file_name