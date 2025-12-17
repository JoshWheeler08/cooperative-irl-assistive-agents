import sys
from environments.kaz.core.fixed_horizon_double_player import FixedHorizonDoubleKAZ

sys.dont_write_bytecode = True


class FixedHorizonRandomKAZ(FixedHorizonDoubleKAZ):

    """ 
        Extension of FixedHorizonDoubleKAZ (fixed-horizon, double-player KAZ environment) where a random policy is used to control 
        both the Owner and Assistant Agent. They can either share the same random policy or have separate ones 
        depending on the boolean flag 'share_random_policy'.
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
        running_in_gym_flag=False,
        env_output = True,
        share_random_policy=False,
        user_seed=None,
        fixed_horizon_num_steps = 1000,
    ):
 
        super(FixedHorizonRandomKAZ, self).__init__(
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
            knight_policy=None,
            running_in_gym_flag=running_in_gym_flag,
            env_output = env_output,
            user_seed=user_seed,
            fixed_horizon_num_steps=fixed_horizon_num_steps,
        )

        # Used to determine whether the archer and knight use the same random policy or separate random policies
        self.share_random_policy = share_random_policy

    # Ignores the argument action - it will select a random action for each player
    def step(self, action):

        # Knight operation
        self.agent = self.knight
        knight_rand_action = self.action_space.sample()
        obs_knight, reward_knight, done_knight, knight_info = self._step(knight_rand_action)

        # Archer operation
        self.agent = self.archer

        if(self.share_random_policy): 
            # use the same action as knight
            archer_rand_action = knight_rand_action
        else:
            # Use different policy 
            archer_rand_action = self.action_space.sample()

        obs_archer, archer_reward, archer_done, archer_info = self._step(archer_rand_action)  


        # Full Game Details
        game_reward = archer_reward + reward_knight
        game_info = {
            'archer_info': archer_info,
            'knight_info' : knight_info,
        }
        game_done = archer_done or done_knight

        
        # Doesn't really matter what observation is returned because the passed in action isn't used.

        return obs_archer, game_reward, game_done, game_info