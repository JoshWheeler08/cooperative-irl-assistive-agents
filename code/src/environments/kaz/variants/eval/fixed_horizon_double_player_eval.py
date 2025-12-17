import sys
from environments.kaz.core.fixed_horizon_double_player import FixedHorizonDoubleKAZ

sys.dont_write_bytecode = True


class EvalFixedHorizonDoubleKAZ(FixedHorizonDoubleKAZ):

    """ 
        Fixed-horizon, double-player version of the KAZ environment where the knight's policy 
        is passed in to allow it to act autonomously. This environment is used for evaluating the overall game performance
        of both agents because it returns their collective reward per step, rather than just the Archer's 
        like other KAZ implementations.
    """

    def __init__(
        self,
        spawn_rate=20,
        max_zombies=10,
        killable_knight=True,
        max_arrows=1,
        killable_archer=True,
        pad_observation=True,
        line_death=False,
        vector_state=True,
        use_typemasks=False,
        transformer=False,
        render_mode=None,
        knight_policy=None,
        running_in_gym_flag=False,
        env_output = True,
        user_seed=None,
        fixed_horizon_num_steps=1000,
    ):
        super(EvalFixedHorizonDoubleKAZ, self).__init__(
            spawn_rate=spawn_rate,
            max_zombies=max_zombies,
            killable_knight=killable_knight,
            max_arrows=max_arrows,
            killable_archer=killable_archer,
            pad_observation=pad_observation,
            line_death=line_death,
            vector_state=vector_state,
            use_typemasks=use_typemasks,
            transformer=transformer,
            render_mode=render_mode,
            knight_policy=knight_policy,
            running_in_gym_flag=running_in_gym_flag,
            env_output = env_output,
            user_seed=user_seed,
            fixed_horizon_num_steps=fixed_horizon_num_steps,
        )


    # Two player step operation
    def step(self, action):
        
        # Observation for knight
        self.agent = self.knight

        if self.knight_policy == None:
            knight_action = self.action_space.sample()
        else: # user has passed in a knight policy
            knight_action, _states = self.knight_policy.predict(self.prev_knight_obs)

        obs_knight, reward_knight, done_knight, info_knight = self._step(knight_action) 

        # Saving knight's observation for determining next action 
        self.prev_knight_obs = obs_knight

        # Observation for archer
        self.agent = self.archer
        obs_archer, reward_archer, done_archer, info_archer = self._step(action)    
        
        # Full Game Details
        game_reward = reward_archer + reward_knight
        game_info = {
            'archer_info': info_archer,
            'knight_info' : info_knight,
        }
        game_done = done_archer or done_knight


        return obs_archer, game_reward, game_done, game_info