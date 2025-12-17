from environments.kaz.core.double_player import DoubleKAZ

from .src import constants as const

class FixedHorizonDoubleKAZ(DoubleKAZ):

    """ 
        Fixed-horizon, double-player version of the KAZ environment where the knight's policy 
        is passed in to allow it to act autonomously.
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

        super(FixedHorizonDoubleKAZ, self).__init__(
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
        )
        
        # Setting the horizon limit & counter to track number of steps taken
        self.fixed_horizon_num_steps = fixed_horizon_num_steps
        self.number_of_steps_counter = 0


    def _step(self, action):
        agent = self.agent # archer or knight
        
        if agent.is_archer:
            # Update counter - remember only the archer is being trained when using this environment 
            # (the knight has already been trained, so not counting its steps)
            self.number_of_steps_counter += 1

        # cumulative rewards from previous iterations should be cleared
        self._cumulative_rewards = 0
        agent.score = 0

        if agent.alive:
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

        # Spawning Zombies at Random Location
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
        
        return self.observe(agent), float(self.rewards), self.done, {}


    # Resets the game
    def reset(self, seed=None):
        self.number_of_steps_counter = 0
        return super().reset(seed)
    

    # Zombie reaches the End of the Screen
    def zombie_endscreen(self, run, zombie_list):
        for zombie in zombie_list:
            if zombie.rect.y > const.SCREEN_HEIGHT - const.ZOMBIE_Y_SPEED:
                # print("Zombies reached end of screen")
                # Remove the zombie
                self.zombie_list.remove(zombie)


    # Checks if the game has ended due to the fixed horizon being reached
    def check_game_end(self):
        self.zombie_endscreen(self.run, self.zombie_list) # Remove any zombies that have passed the threshold
        self.run = not(self.fixed_horizon_num_steps == self.number_of_steps_counter) # game only finishes when the horizon has been reached