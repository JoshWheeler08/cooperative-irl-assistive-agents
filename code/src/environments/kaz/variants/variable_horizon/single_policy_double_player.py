import sys
from common.constants import IGNORE_HORIZON
from environments.kaz.variants.variable_horizon.full_obs_double_player import FullObsDoubleKAZ

sys.dont_write_bytecode = True


class SingleRLPolicyDoubleKAZ(FullObsDoubleKAZ):
    """ 
        Extension of the FullObsDoubleKAZ environment, which is used to train a SINGLE RL policy for controlling both the knight and archer agents.
        In order to do this, the RL agent is given a full game observation from the perspective of the archer - arbritary choice, but makes sense since the purpose
        of this work is to research the Assistant agent's behaviour. Furthermore, the rewards from both the knight and archer agents are returned together 
        with the hope that the RL policy learns to maximise their collective total reward. Both agents take the same action.
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
        knight_policy = None,
        user_seed=None,
        fixed_horizon_num_steps=IGNORE_HORIZON,
    ):  

        super(SingleRLPolicyDoubleKAZ, self).__init__(
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
            running_in_gym_flag=running_in_gym_flag,
            env_output=env_output,
            knight_policy=knight_policy,
            user_seed=user_seed,
        )

    # Both agents take the same action 
    def step(self, action):

        # Knight step
        self.agent = self.knight
        _, knight_reward, knight_done, knight_info = self._step(action)

        # Archer step
        self.agent = self.archer
        full_obs, archer_reward, archer_done, archer_info = self._step(action)
        
        # Full Game Details
        game_reward = archer_reward + knight_reward
        game_info = {
            'archer_info': archer_info,
            'knight_info' : knight_info,
        }
        game_done = archer_done or knight_done
        
        return full_obs, game_reward, game_done, game_info