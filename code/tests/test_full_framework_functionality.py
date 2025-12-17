""" 
    Program to test all the API commands in my *Owner* and *Assistant* agent implementations 
    by running through the experiment pipeline, which consists of:  

        * Registering the KAZ environments with OpenAI Gym library
        * Creating the Owner Agent
        * Training the Owner Agent
        * Evaluating the Owner Agent
        * Playing a game with the Owner Agent (not necessary for experiment but included to test functionality)
        * Saving the Owner Agent's Policy
        * Loading the Owner Agent's Policy
        * Creating the Assistant Agent
        * Training the Assistant Agent
        * Evaluating the Assistant Agent's RL policy (for learning single player behaviour)
        * Evaluating the Assistant Agent's internal model (for intention recognition)
        * Evaluating the Assistant Agent's Final Policy (RL + IRL)
        * Saving the Assistant Policy
        * Loading the Assistant Policy
        * Playing a game with the Assistant's RL policy (not necessary for experiment but included to test functionality)
        * Playing a game with the Assistant's learned Internal Model of the Owner agent (knight)
        * Playing a game with the Assistant's Final policy, where the `knight` is also acting in the game
"""

import os
import sys

# Add src directory to path so imports work
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gym.envs.registration import register

# My implementations
from wrappers.kaz_training_wrapper import KAZTrainingWrapper
from agents.helpers.env_config import EnvObject
from agents.assistant import Assistant
from agents.owner import Owner


def main(
    max_episode_steps,
    total_timesteps,
    n_envs_per_training,
    irl_alg,
    env_output=True,
    owner_agent_policy="PPO",
    owner_agent_policy_type="MlpPolicy",
    final_assistant_policy="PPO",
    final_assistant_policy_type="MlpPolicy",
    assistant_rl_policy="A2C",
    assistant_rl_policy_type="MlpPolicy",
    use_fixed_horizon=False,
    fixed_horizon_num_steps=1000,
):

    # Selecting which KAZ environments to use for this API test (fixed or variable horizon)
    entry_point_single = (
        "envs.kaz_core.fixed_horizon_single_player:FixedHorizonSingleKAZ"
        if use_fixed_horizon
        else "envs.kaz_core.single_player:SingleKAZ"
    )
    entry_point_double = (
        "envs.kaz_core.fixed_horizon_double_player:FixedHorizonDoubleKAZ"
        if use_fixed_horizon
        else "envs.kaz_core.double_player:DoubleKAZ"
    )

    # Register Environments
    register(
        id="SingleKAZ-v0",
        entry_point=entry_point_single,
        max_episode_steps=max_episode_steps,
    )

    register(
        id="DoubleKAZ-v0",
        entry_point=entry_point_double,
        max_episode_steps=max_episode_steps,
    )

    # Create Owner
    owner_agent = Owner()

    # ----- Train Owner -----

    # on SingleKAZ with Knight
    knight_env_kwargs = {
        "type_of_player": "knight",
        "env_output": env_output,
    }

    if use_fixed_horizon:
        # setting horizon limit
        knight_env_kwargs["fixed_horizon_num_steps"] = fixed_horizon_num_steps

    # EnvObject describes the environment to be used for training the knight
    knight_env_obj = EnvObject(
        env_id="SingleKAZ-v0",
        n_envs=n_envs_per_training,
        total_timesteps=total_timesteps,
        env_kwargs=knight_env_kwargs,
    )

    # Train Owner agent
    print("[Operation] TRAINING THE OWNER AGENT ")

    owner_agent.train_rl(
        knight_env_obj,
        policy=owner_agent_policy,
        policy_type=owner_agent_policy_type,
    )

    # Evaluate Owner Agent
    print("[Operation] Quickly evaluating RL model:")
    mean_reward, mean_std, mean_episode_length = owner_agent.evaluate_policy(
        knight_env_obj, n_eval_episodes=10, csv_path="./test_framework/evaluation/owner/"
    )
    print(
        f"Mean reward = {mean_reward}, Std reward  = {mean_std}, Mean episode length = {mean_episode_length}"
    )

    # Play Game with Owner Policy
    owner_agent.play_game_with_rl_policy(knight_env_obj)

    # Save Policy for Owner Agent
    owner_agent_policy_path = (
        f"./test_framework/learned_policies/owner_agent_experiment_1/"
    )
    owner_agent.save_policy(
        path=owner_agent_policy_path, policy_type=owner_agent_policy
    )

    # Load Owner Agent
    del owner_agent
    owner_agent_reborn = Owner()
    owner_agent_reborn.load_policy(
        path=f"{owner_agent_policy_path}", policy_type=owner_agent_policy
    )

    # Create Assistant Agent
    assistant_agent = Assistant()

    # ----- Train Assistant Agent -----

    # Creating archer environment
    archer_env_kwargs = {
        "type_of_player": "archer",
        "env_output": env_output,
    }

    if use_fixed_horizon:
        archer_env_kwargs["fixed_horizon_num_steps"] = fixed_horizon_num_steps

    # Describing the environment to be used for training the archer's RL policy
    archer_env_obj = EnvObject(
        env_id="SingleKAZ-v0",
        n_envs=n_envs_per_training,
        total_timesteps=total_timesteps,
        env_kwargs=archer_env_kwargs,
    )

    # Creating Archer + Knight KAZ Environment (2-player) for training the Archer's final policy
    combiner_env_kwargs = {
        "knight_policy": owner_agent_reborn.policy,  # the FIXED owner policy will control the knight during assistant's training
        "env_output": env_output,
    }

    if use_fixed_horizon:
        combiner_env_kwargs["fixed_horizon_num_steps"] = fixed_horizon_num_steps

    # Setting up two player KAZ environment
    combiner_env_obj = EnvObject(
        env_id="DoubleKAZ-v0",
        n_envs=n_envs_per_training,
        total_timesteps=total_timesteps,
        env_kwargs=combiner_env_kwargs,
        wrapper_class=KAZTrainingWrapper,
    )

    # Train Assistant
    print("[Operation] TRAINING THE ASSISTANT AGENT")
    assistant_agent.train(
        assistant_env_obj=archer_env_obj,
        internal_model_env_obj=knight_env_obj,
        combiner_env_obj=combiner_env_obj,
        expert_owner_agent=owner_agent_reborn,
        assistant_rl_policy=assistant_rl_policy,
        assistant_rl_policy_type=assistant_rl_policy_type,
        combiner_rl_policy=final_assistant_policy,
        combiner_rl_policy_type=final_assistant_policy_type,
        irl_alg=irl_alg,
    )

    # EVALUATION

    # Evaluating Assistant's Rl policy
    print("[Operation] Quickly evaluating Assistant RL model:")
    mean_reward, mean_std, mean_episode_length = assistant_agent.evaluate_rl_policy(
        archer_env_obj, n_eval_episodes=10, csv_path="./test_framework/evaluation/assistant_rl/"
    )
    print(
        f"Mean reward = {mean_reward}, Std reward  = {mean_std}, Mean episode length = {mean_episode_length}"
    )

    # Evaluating Assistant's Internal Model of Owner agent
    print("[Operation] Quickly evaluating Assistant Internal model:")
    (
        mean_reward,
        mean_std,
        mean_episode_length,
    ) = assistant_agent.evaluate_internal_model_policy(
        knight_env_obj, n_eval_episodes=10, csv_path="./test_framework/evaluation/assistant_irl/"
    )
    print(
        f"Mean reward = {mean_reward}, Std reward  = {mean_std}, Mean episode length = {mean_episode_length}"
    )

    # Evaluating Assistant's final policy
    print("[Operation] Quickly evaluating Assistant Final Policy:")
    mean_reward, mean_std, mean_episode_length = assistant_agent.evaluate_policy(
        combiner_env_obj, n_eval_episodes=10, csv_path="./test_framework/evaluation/assistant/"
    )
    print(
        f"Mean reward = {mean_reward}, Std reward  = {mean_std}, Mean episode length = {mean_episode_length}"
    )

    # Save Assistant policy
    assistant_agent_policy_path = (
        f"./test_framework/learned_policies/assistant_agent_experiment_1/"
    )
    assistant_agent.save_policy(
        path=assistant_agent_policy_path,
        rl_policy_type=assistant_rl_policy,
        final_policy_type=final_assistant_policy,
    )

    # Load Assistant Policy
    del assistant_agent
    assistant_agent_reborn = Assistant()

    assistant_agent_reborn.load_policy(
        path=assistant_agent_policy_path,
        final_policy_type=final_assistant_policy,
        rl_policy_type=assistant_rl_policy,
        irl_alg=irl_alg,
        irl_policy_type=None,
    )

    # Play Game with Assistant RL Policy
    print("[Operation] Play game with Assistant RL Policy")
    assistant_agent_reborn.play_game_with_rl_policy(env_obj=archer_env_obj)

    # Play Game with Internal Model Policy
    print("[Operation] Play game with Assistant Internal Model Policy")
    assistant_agent_reborn.play_game_with_internal_model_policy(env_obj=knight_env_obj)

    # Play Game with Final Policy
    game_env_obj = EnvObject(
        env_id="DoubleKAZ-v0",
        total_timesteps=total_timesteps,
        env_kwargs=combiner_env_kwargs,
        wrapper_class=KAZTrainingWrapper,
    )

    print("[Operation] Play game with Assistant Final Policy")
    assistant_agent_reborn.play_game_with_final_policy(env_obj=game_env_obj)


if __name__ == "__main__":

    # Passing in some example hyperparameter values
    main(
        max_episode_steps=None,
        total_timesteps=10,
        n_envs_per_training=4,
        env_output=False,
        irl_alg="AIRL",
        owner_agent_policy="PPO",
        owner_agent_policy_type="MlpPolicy",
        final_assistant_policy="PPO",
        final_assistant_policy_type="MlpPolicy",
        assistant_rl_policy="PPO",
        assistant_rl_policy_type="MlpPolicy",
        use_fixed_horizon=True,
        fixed_horizon_num_steps=100,
    )
