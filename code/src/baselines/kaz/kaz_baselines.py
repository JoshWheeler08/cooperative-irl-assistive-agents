"""
    This is a file storing my KAZ baseline test implementations.

"""

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

from common.common import *
from agents.owner import Owner
from common.constants import IGNORE_HORIZON


# RL Algorithm Implementations
supported_policy_types = Owner().supported_rl_algs


# Single version of KAZ environment (SingleKAZ) used to train the Archer and Knight in isolation.
# Then, it puts them together in the Double player environment (DoubleKAZ) to see how they perform
def test_separate_policies(
        total_timesteps=10,
        n_envs=4, 
        training_env_id="SingleKAZ-v0",
        eval_env_id="EvalDoubleKAZ-v0", 
        n_eval_episodes=10,
        model_path=None,
        policy_type="PPO",
        csv_path="./monitor_dir/test_separate_policies/",
        wandb_run=None,
        user_seed=None,
        fixed_horizon_num_steps = IGNORE_HORIZON,
        max_zombies = 10,
    ):

    """ Both players have their own RL policy """

    if not(policy_type in supported_policy_types):
        print("[Error] Invalid policy type - aborting baseline test")
        return None

    # Train PPO Algorithms
    knight_log_folder = f"{csv_path}knight"
    archer_log_foler = f"{csv_path}archer" 
    type_of_players = {
        "knight": configure(knight_log_folder, ["csv", "stdout"]),
        "archer": configure(archer_log_foler, ["csv", "stdout"]),
    }

    models = {}

    # Train Separate PPO Algorithm for Owner and Assistant Agent
    for player, logger in type_of_players.items():
        try:
            # Set up environment args
            env_kwargs = {
                "type_of_player" : player,
                "env_output" : False,
                "user_seed":user_seed,
                "fixed_horizon_num_steps": fixed_horizon_num_steps,
                "max_zombies": max_zombies,
            }

            # Create vectorised environment
            env = make_vec_env(
                training_env_id, 
                n_envs=n_envs, 
                env_kwargs=env_kwargs,
            ) # Parallel environments

            # Train SB3 Agent
            model = supported_policy_types[policy_type](
                "MlpPolicy", 
                env, 
                verbose=1,
                seed = user_seed,
            )

            # Set logger to store metrics in CSV file
            model.set_logger(logger)
        
            model.learn(
                total_timesteps=total_timesteps,
            )

            models[player] = model

            # Tidy up 
            env.close()

        except Exception as e:
            print(f"[Error] Failed to train player '{player}' in baselines test (test_separate_policies) - aborting")
            return None
        
    # Evaluate policies together using DoubleKAZ

    env_kwargs = {
        "knight_policy": models["knight"].policy, # the FIXED owner policy will control the knight during assistant training
        "env_output":False,
        "user_seed": user_seed,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    # Making evaluation environment
    eval_env = make_vec_env(
        env_id=eval_env_id, 
        n_envs=n_envs, 
        env_kwargs=env_kwargs,
    ) 

    episode_rewards, episode_lengths = evaluate_policy(
        models["archer"], 
        eval_env, 
        n_eval_episodes,
        return_episode_rewards=True,
    )

    # Process evaluation data
    mean_reward, std_reward, mean_episode_length, csv_file_name = process_evaluation_results(episode_rewards, episode_lengths, csv_path)

    # Output evaluation results
    print(f"Mean reward = {mean_reward}, Standard deviation in reward = {std_reward}, Mean episode length = {mean_episode_length}")

    # Save models locally for reproducibility
    if not(model_path == None):
        for player_name, policy in models.items():
            new_path = f"{model_path}_{player_name}_{policy_type}"
            policy.save(new_path)

        print("[Success] Policies saved")

    # Log results to WandB
    if wandb_run is not None:
        log_results_to_wandb({
            f"knight":f"{knight_log_folder}/progress.csv",
            f"archer":f"{archer_log_foler}/progress.csv",
            f"evaluation":f"{csv_path}{csv_file_name}",
            },
            wandb_run=wandb_run,
        )

    return [mean_reward, std_reward, mean_episode_length]


# Both Players follow different random policies
def test_random_policies(
        n_envs=4, 
        eval_env_id="EvalRandomKAZ-v0",
        n_eval_episodes=10,
        path_to_expert=None,
        policy_type="PPO",
        env_kwargs=None,
        csv_path="./monitor_dir/test_random_policies/",
        wandb_run = None,
        user_seed=None,
        fixed_horizon_num_steps=IGNORE_HORIZON,
        max_zombies = 10,
    ):

    if env_kwargs == None:
        env_kwargs = {
            "env_output":False,
            "user_seed":user_seed,
            "share_random_policy":False,
            "fixed_horizon_num_steps": fixed_horizon_num_steps,
            "max_zombies": max_zombies,
        }

    # Create evaluation env
    eval_env = make_vec_env(
        env_id=eval_env_id, 
        n_envs=n_envs, 
        env_kwargs=env_kwargs,
    )

    # Need some FILLER policy for evaluate_policy - note that this learned policy's output is NOT used by the environment (EvalRandomKAZ)
    expert_archer = supported_policy_types[policy_type].load(path_to_expert)

    # Start evaluation 
    episode_rewards, episode_lengths = evaluate_policy(
        expert_archer,
        eval_env, 
        n_eval_episodes,
        return_episode_rewards=True,
    )

    # Process evaluation data
    mean_reward, std_reward, mean_episode_length, csv_file_name = process_evaluation_results(episode_rewards, episode_lengths, csv_path)

    # Output evaluation results
    print(f"Mean reward = {mean_reward}, Standard deviation in reward = {std_reward}, Mean episode length = {mean_episode_length}")

    # Tidy up
    eval_env.close()

    # Log results to WandB
    if wandb_run is not None:
        log_results_to_wandb({
                f"evaluation":f"{csv_path}{csv_file_name}",
            },
            wandb_run=wandb_run,
        )

    # Return results
    return [mean_reward, std_reward, mean_episode_length]


# Both Players follow the same random policy
def test_same_random_policy(
        n_envs=4, 
        eval_env_id="EvalRandomKAZ-v0", 
        n_eval_episodes=10,
        path_to_expert=None,
        policy_type="PPO",
        env_kwargs=None,
        csv_path="./monitor_dir/test_same_random_policy/",
        wandb_run = None,
        user_seed=None,
        fixed_horizon_num_steps = IGNORE_HORIZON,
        max_zombies = 10,
    ):

    env_kwargs = {
        "env_output":False,
        "share_random_policy":True, # important flag
        "user_seed": user_seed,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    return test_random_policies(
            n_envs=n_envs, 
            eval_env_id=eval_env_id, 
            n_eval_episodes=n_eval_episodes, 
            path_to_expert=path_to_expert, 
            policy_type=policy_type, 
            env_kwargs=env_kwargs, 
            csv_path=csv_path, 
            wandb_run=wandb_run,
            user_seed=user_seed,
            fixed_horizon_num_steps=fixed_horizon_num_steps,
            max_zombies=max_zombies,
        )


# Knight learns policy on single-player KAZ, Archer learns policy on double-player KAZ while knight acts 
# Note that the Archer is only given a LOCAL observation per timestep (in constrast to the next baseline test)
def test_train_on_different_envs(
    single_player_policy_type="PPO",
    double_player_policy_type="PPO",
    single_player_env_id="SingleKAZ-v0",
    double_player_env_id="DoubleKAZ-v0",
    eval_env_id="EvalDoubleKAZ-v0",
    n_envs=4,
    total_timesteps=1000,
    n_eval_episodes=10,
    model_path=None,
    csv_path="./monitor_dir/test_train_on_different_envs/",
    wandb_run=None,
    user_seed=None,
    fixed_horizon_num_steps = IGNORE_HORIZON,
    max_zombies = 10,
):
    """ Knight learns RL policy on SingleKAZ with Archer, and Archer learns RL policy on DoubleKAZ - while the Knight is simultaneously acting in the environment"""

    if not(single_player_policy_type in supported_policy_types) or not(double_player_policy_type in supported_policy_types):
        print("[Error] Invalid policy type - aborting baseline test")
        return None

    # ---- Train RL Algorithms ----

    # Train Single Player first

    # Set up environment args
    knight_env_kwargs = {
        "type_of_player" : "knight",
        "env_output" : False,
        "user_seed": user_seed,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    # Create vectorised environment
    env = make_vec_env(single_player_env_id, n_envs=n_envs, env_kwargs=knight_env_kwargs) # Parallel environments

    # Train SB3 Agent
    knight_model = supported_policy_types[single_player_policy_type](
        "MlpPolicy", 
        env, 
        verbose=1,
        seed = user_seed,
    )

    # Set up SB3 Logger (CSV file)
    knight_log_folder = f"{csv_path}knight"
    knight_logger = configure(knight_log_folder, ["csv", "stdout"])
    knight_model.set_logger(knight_logger)

    knight_model.learn(
        total_timesteps=total_timesteps
    )

    # Tidy up 
    env.close()
        
    # ---- Train Archer in 2-Player Env ----

    # Set up environment args
    archer_env_kwargs = {
        "knight_policy": knight_model.policy,
        "env_output" : False,
        "user_seed": user_seed,
        "fixed_horizon_num_steps": fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    # Create vectorised environment
    archer_env = make_vec_env(
        double_player_env_id, 
        n_envs=n_envs, 
        env_kwargs=archer_env_kwargs,
    ) # Parallel environments

    # Train Archer SB3 Agent
    archer_model = supported_policy_types[double_player_policy_type](
        "MlpPolicy", 
        archer_env, 
        verbose=1,
        seed=user_seed,
    )

    # Set up SB3 Logger (CSV file)
    archer_log_folder = f"{csv_path}archer"
    archer_logger = configure(archer_log_folder, ["csv", "stdout"])
    archer_model.set_logger(archer_logger)

    archer_model.learn(
        total_timesteps=total_timesteps
    )


    # Evaluate policies together using DoubleKAZ
    eval_env = make_vec_env(
        env_id=eval_env_id, 
        n_envs=n_envs, 
        env_kwargs=archer_env_kwargs,
    )

    episode_rewards, episode_lengths = evaluate_policy(
        archer_model, 
        eval_env,
        n_eval_episodes,
        return_episode_rewards=True,
    )
    
    # Process evaluation data
    mean_reward, std_reward, mean_episode_length, csv_file_name = process_evaluation_results(episode_rewards, episode_lengths, csv_path)

    # Tidy up 
    archer_env.close()

    # Output evaluation results
    print(f"Mean reward = {mean_reward}, Standard deviation in reward = {std_reward}, Mean episode length = {mean_episode_length}")

    # Save models for reproducibility
    if not(model_path == None):

        # Save Knight Policy
        knight_new_path = f"{model_path}_knight_{single_player_policy_type}"
        knight_model.save(knight_new_path)

        # Save Archer Policy
        archer_new_path = f"{model_path}_archer_{double_player_policy_type}"
        archer_model.save(archer_new_path)

        print("[Success] Policies saved")

    # Log results to WandB
    if wandb_run is not None:
        log_results_to_wandb({
                f"knight":f"{knight_log_folder}/progress.csv",
                f"archer":f"{archer_log_folder}/progress.csv",
                f"evaluation":f"{csv_path}{csv_file_name}",
            },
            wandb_run = wandb_run,
        )

    return [mean_reward, std_reward, mean_episode_length]


# Repeat of test_train_on_different_envs, but the Archer is given a FULL observation of the game per timestep
def test_train_on_different_envs_full_observation(
    single_player_policy_type="PPO",
    double_player_policy_type="PPO",
    single_player_env_id="SingleKAZ-v0",
    double_player_env_id="FullObsDoubleKAZ-v0",
    eval_env_id="EvalFullObsDoubleKAZ-v0",
    n_envs=4,
    total_timesteps=1000,
    n_eval_episodes=10,
    model_path=None,
    csv_path="./monitor_dir/test_train_on_different_envs_full_observation/",
    wandb_run=None,
    user_seed = None,
    fixed_horizon_num_steps = IGNORE_HORIZON,
    max_zombies = 10,
):
    return test_train_on_different_envs(
        single_player_policy_type=single_player_policy_type,
        double_player_policy_type=double_player_policy_type,
        single_player_env_id=single_player_env_id,
        double_player_env_id=double_player_env_id,
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        n_eval_episodes=n_eval_episodes,
        model_path=model_path,
        csv_path=csv_path,
        wandb_run=wandb_run,
        user_seed=user_seed,
        eval_env_id=eval_env_id,
        fixed_horizon_num_steps=fixed_horizon_num_steps,
        max_zombies=max_zombies,
    )


# Learns a single RL policy for controlling both AGENTS given the full game observation per timestep
def test_single_rl_policy_full_observation(
    env_id="SingleRLPolicyDoubleKAZ-v0",
    n_envs=4,
    policy_type="PPO",
    total_timesteps=1000,
    n_eval_episodes=10,
    model_path=None,
    csv_path="./monitor_dir/test_single_rl_policy_full_observation/",
    wandb_run=None,
    user_seed = None,
    fixed_horizon_num_steps = IGNORE_HORIZON,
    max_zombies = 10,
):
    
    """  Learn a single RL policy based on the full game observation """

    if not(policy_type in supported_policy_types):
        print("[Error] Invalid policy type - aborting baseline test")
        return None

    # Set up environment args
    env_kwargs = {
        "env_output":False,
        "user_seed":user_seed,
        "fixed_horizon_num_steps" : fixed_horizon_num_steps,
        "max_zombies": max_zombies,
    }

    try:
        # Create vectorised environment
        env = make_vec_env(env_id, n_envs=n_envs, env_kwargs=env_kwargs)

        
        # Train SB3 Agent
        full_model = supported_policy_types[policy_type](
            "MlpPolicy", 
            env, 
            verbose=1,
            seed=user_seed,
        )

        # Set up SB3 logger
        single_policy_log_folder = f"{csv_path}single_policy"
        single_policy_logger = configure(single_policy_log_folder, ["csv", "stdout"])

        full_model.set_logger(single_policy_logger)

        full_model.learn(total_timesteps=total_timesteps)

    except Exception as e:
        print("[Error] Failed to train agent in baseline test (test_single_rl_policy_full_observation) - aborting ")
        return None

    # Performing evaluation

    episode_rewards, episode_lengths = evaluate_policy(
        full_model, 
        env,
        n_eval_episodes,
        return_episode_rewards=True,
    )

    # Process evaluation data
    mean_reward, std_reward, mean_episode_length, csv_file_name = process_evaluation_results(episode_rewards, episode_lengths, csv_path)

    # Tidy up
    env.close()

    # Output evaluation results
    print(f"Mean reward = {mean_reward}, Standard deviation in reward = {std_reward}, Mean episode length = {mean_episode_length}")

    # Save model for reproducibility
    if not(model_path == None):
        # Save Joint Policy
        new_path = f"{model_path}_joint_policy_{policy_type}"
        full_model.save(new_path)

        print("[Success] Policy saved")

    # Log results to WandB
    if wandb_run is not None:
        log_results_to_wandb({
                f"single_policy":f"{single_policy_log_folder}/progress.csv",
                f"evaluation":f"{csv_path}{csv_file_name}",
            },
            wandb_run=wandb_run,
        )

    return [mean_reward, std_reward, mean_episode_length]