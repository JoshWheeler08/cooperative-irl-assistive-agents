import sys
import numpy as np

from common.constants import IGNORE_HORIZON
from gym import spaces
from environments.kaz.core.double_player import DoubleKAZ

sys.dont_write_bytecode = True


class FullObsDoubleKAZ(DoubleKAZ):
    """ 
        Variable-horizon, double-player KAZ environment where the archer's per-step observation has been changed 
        from a local to global scope. 
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
        fixed_horizon_num_steps=IGNORE_HORIZON,
    ):  

        super(FullObsDoubleKAZ, self).__init__(
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

        # Local variables
        num_archers = 1
        num_knights = 1

        # For full game observation
        self.num_tracked = (
            num_archers + num_knights + max_zombies + num_knights + max_arrows
        )

        # Update observation space
        shape = (
            [512, 512, 3]
            if not self.vector_state
            else [1, (self.num_tracked + 1) * (self.vector_width + 1)]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


    def _step(self, action):
        agent = self.agent

        # cumulative rewards from previous iterations should be cleared
        self._cumulative_rewards = 0
        agent.score = 0

        # this is... so whacky... but all actions here are index with 1 so... ok -> PettingZoo comment
        action = action + 1
        out_of_bounds = agent.update(action)

        # check for out of bounds death
        if self.line_death and out_of_bounds:
            if self.env_output:
                print(f"{agent.agent_name} went out of bounds")
            self.agent.alive = False
            self.agent.weapons.empty()

        # actuate the weapon if necessary
        self.action_weapon(action, agent)

        # Update the weapons
        self.update_weapons(agent)

        if agent.is_archer:
            # Archer kills the Zombie
            self.arrow_hit()

            # Zombie Kills the Archer
            if self.killable_archer:
                self.zombie_hit_archer()
        else:
            # Knight kills the Zombie
            self.sword_hit()

            # Zombie Kills the Knight
            if self.killable_knight:
                self.zombit_hit_knight()

        # update some zombies
        for zombie in self.zombie_list:
            zombie.update()

        # Spawning Zombies at Random Location at every 100 iterations
        self.spawn_zombie()

        self.draw()

        self.check_game_end()
        self.frames += 1

        self.done = not self.run

        self.rewards = 0
        self.rewards = agent.score

        self._cumulative_rewards += self.rewards

        if self.render_mode == "human":
            self.render()
        
        # To prove that 2 different observation spaces are being used (1, 65) and (1, 75)
        # x = self.observe(agent, agent.is_archer)
        # print(x.shape)

        return self.observe(agent, agent.is_archer), float(self.rewards), self.done, {}

    # Resets the game
    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed=seed)
        self.has_reset = True
        self.rewards = 0
        self._cumulative_rewards = 0
        self.done = False
        self.reinit()

        if self.render_mode == "human":
            self.render()

        # Must include self.infos ({}) in return line if using gym alone, otherwise remove for cleanRL
        
        if self.vector_state:
            # Prepare observation

            # Get observations for each player
            obs_archer = self.observe(self.archer, True)
            self.prev_knight_obs = self.observe(self.knight)

            # Reshape archer observation to fit expected observation space
            obs = obs_archer.reshape([1, (self.num_tracked + 1) * (self.vector_width + 1)])

        else:
            # Just pixel data, will be the same for either agent because its a snapshot of the entire game
            obs = self.observe(self.archer)


        # Gym vs SB3 Compatibility
        if self.running_in_gym_flag:
            return obs, {}

        else:
            return obs
