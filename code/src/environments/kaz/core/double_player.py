import os
import sys
from itertools import repeat

import gymnasium
import gym

from gym import spaces

import numpy as np
import pygame
import pygame.gfxdraw
from gymnasium.utils import EzPickle, seeding

from common.constants import IGNORE_HORIZON
from .src import constants as const
from .src.img import get_image
from .src.players import Archer, Knight
from .src.weapons import Arrow, Sword
from .src.zombie import Zombie

sys.dont_write_bytecode = True


class DoubleKAZ(gym.Env, EzPickle):
    """ 
        Variable-horizon, double-player version of the KAZ environment where the knight's policy 
        is passed in to allow it to act autonomously 
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "knights_archers_zombies_v10",
        "render_fps": const.FPS,
        "has_manual_policy": True,
    }


#    --------- Core OpenAI Gym Methods ------------

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
        # Fixing the allowed number of players to 2
        num_archers = 1
        num_knights = 1

        EzPickle.__init__(
            self,
            spawn_rate,
            max_zombies,
            max_arrows,
            killable_archer,
            killable_knight,
            pad_observation,
            line_death,
            vector_state,
            use_typemasks,
            transformer,
            render_mode,
            knight_policy,
            num_archers,
            num_knights,
            user_seed,
        )

        super(DoubleKAZ, self).__init__()

        # variable state space
        self.transformer = transformer

        # whether we want RGB state or vector state
        self.vector_state = vector_state
        # agents + zombies + weapons

        # For assistant initial RL policy
        self.num_tracked = (
            num_archers + max_zombies + max_arrows # ([num_archers + max_zombies + max_arrows + 1] x 5) - observation dimensions
        )

        # For full game observation
        self.num_tracked_full_obs = (
            num_archers + num_knights + max_zombies + num_knights + max_arrows
        )

        # So that the knight-trained policy receives the same OBSERVATION it was trained on!
        self.num_tracked_knight = (
            num_knights + num_knights + max_zombies # ([num_knights + num_knights + max_zombies + 1] x 5) - observation dimensions
        )

        self.use_typemasks = True if transformer else use_typemasks
        self.typemask_width = 6
        self.vector_width = 4 + self.typemask_width if use_typemasks else 4

        # Game Status
        self.frames = 0
        self.closed = False
        self.has_reset = False
        self.render_mode = render_mode
        self.render_on = False
        self.env_output=env_output

        # Game Constants
        self.seed(seed=user_seed)
        self.spawn_rate = spawn_rate
        self.pad_observation = pad_observation
        self.killable_knight = killable_knight
        self.killable_archer = killable_archer
        self.max_arrows = max_arrows
        self.line_death = line_death
        self.max_zombies = max_zombies
        self.num_archers = num_archers
        self.num_knights = num_knights
        self.running_in_gym_flag = running_in_gym_flag

        # Rewards
        self._cumulative_rewards = 0
        self.rewards = 0

        # Knight Policy
        self.knight_policy = knight_policy
        self.prev_knight_obs = None

        shape = (
            [512, 512, 3]
            if not self.vector_state
            else [1, (self.num_tracked + 1) * (self.vector_width + 1)]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        
        self.action_space = spaces.Discrete(6)

        # Not sure about below code from PettingZoo
        shape = (
            [const.SCREEN_HEIGHT, const.SCREEN_WIDTH, 3]
            if not self.vector_state
            else [self.num_tracked, self.vector_width]
        )
        low = 0 if not self.vector_state else -1.0
        high = 255 if not self.vector_state else 1.0
        dtype = np.uint8 if not self.vector_state else np.float64
        self.state_space = spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=dtype,
        )
        #################

        # Initializing Pygame
        pygame.init()
        # self.WINDOW = pygame.display.set_mode([self.WIDTH, self.HEIGHT])
        self.WINDOW = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        pygame.display.set_caption("Knights, Archers, Zombies")
        self.left_wall = get_image(os.path.join("img", "left_wall.png"))
        self.right_wall = get_image(os.path.join("img", "right_wall.png"))
        self.right_wall_rect = self.right_wall.get_rect()
        self.right_wall_rect.left = const.SCREEN_WIDTH - self.right_wall_rect.width
        self.floor_patch1 = get_image(os.path.join("img", "patch1.png"))
        self.floor_patch2 = get_image(os.path.join("img", "patch2.png"))
        self.floor_patch3 = get_image(os.path.join("img", "patch3.png"))
        self.floor_patch4 = get_image(os.path.join("img", "patch4.png"))

        self.reinit()

    # Allows both agents to perform an action in the environment
    def step(self, action):
        
        # Observation for knight
        self.agent = self.knight

        if self.knight_policy == None:
            knight_action = self.action_space.sample()
        else: # user has passed in a knight policy
            knight_action, _states = self.knight_policy.predict(self.prev_knight_obs)

        # Performing knight's action
        obs_knight, reward_knight, done_knight, info_knight = self._step(knight_action)
        # Knight's policy has already been trained, so don't need to return its reward at the end of the function.

        # Storing knight's observation for when it needs to take another step
        self.prev_knight_obs = obs_knight

        # Observation for archer
        self.agent = self.archer
        obs_archer, reward_archer, done_archer, info_archer = self._step(action)    
        
        # Only returning information about the archer because the knight is assumed to be part of the environment, which CANNOT be controlled
        return obs_archer, reward_archer, done_archer, info_archer


    # Actually performs the agent's action in the game environment
    def _step(self, action):
        agent = self.agent # knight or archer

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
            # check if Archer kills a Zombie
            self.arrow_hit()

            # Zombie Kills the Archer
            if self.killable_archer:
                self.zombie_hit_archer()
        else:
            # check if Knight kills a Zombie
            self.sword_hit()

            # Zombie Kills the Knight
            if self.killable_knight:
                self.zombit_hit_knight()

        # update some zombies
        for zombie in self.zombie_list:
            zombie.update()

        # Spawn Zombies if possible
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


    # Resets the game if either termination condition met
    def reset(self, seed=None):
        #super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed=seed)
        self.has_reset = True
        self.rewards = 0
        self._cumulative_rewards = 0
        self.done = False
        self.reinit()

        if self.render_mode == "human":
            self.render()
        
        if self.vector_state:
            # Prepare observation

            # Get observations for each player
            obs_archer = self.observe(self.archer)
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

    # Renders the game environment for human
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if not self.render_on and self.render_mode == "human":
            # sets self.render_on to true and initializes display
            self.enable_render()

        observation = np.array(pygame.surfarray.pixels3d(self.WINDOW))

        if self.render_mode == "human":
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    # Closes the game 
    def close(self):
        if not self.closed:
            self.closed = True
            if self.render_on:
                # self.WINDOW = pygame.display.set_mode([const.SCREEN_WIDTH, const.SCREEN_HEIGHT])
                self.WINDOW = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
                self.render_on = False
                pygame.event.pump()
                pygame.display.quit()


# -------------- Additional Game Methods ------------------

    # Returns a full observation of the game from the archer's perspective (used in making the Assistant final RL policy)
    def get_full_observation(self):
        return self.observe(self.archer, full_obs=True)


    # Gets the knight's last observation which can be used for deciding its next action
    def get_knight_observation(self):
        return self.prev_knight_obs

    # Returns the observation space of the KAZ environment
    def observation_space(self):
        return self.observation_space

    # Returns the action space of the KAZ environment
    def action_space(self):
        return self.action_space

    # Calculates a random seed
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


    # Spawn Zombie 
    def spawn_zombie(self):
        if len(self.zombie_list) < self.max_zombies:
            self.zombie_spawn_rate += 1 
            zombie = Zombie(self.np_random) # creating new zombie

            if self.zombie_spawn_rate >= self.spawn_rate:
                zombie.rect.x = self.np_random.integers(0, const.SCREEN_WIDTH) # spawn in random location on X-axis
                zombie.rect.y = 5

                self.zombie_list.add(zombie)
                self.zombie_spawn_rate = 0

    # actuate weapons
    def action_weapon(self, action, agent):
        if action == 5:
            if agent.is_knight:
                if agent.weapon_timeout > const.SWORD_TIMEOUT:
                    # make sure that the current knight doesn't have a sword already
                    if len(agent.weapons) == 0:
                        agent.weapons.add(Sword(agent))

            if agent.is_archer:
                if agent.weapon_timeout > const.ARROW_TIMEOUT:
                    # make sure that the screen has less arrows than allowable
                    if self.num_active_arrows < self.max_arrows:
                        agent.weapons.add(Arrow(agent))


    # move weapons
    def update_weapons(self, agent):
        for weapon in list(agent.weapons):
            weapon.update()

            if not weapon.is_active:
                agent.weapons.remove(weapon)


    @property
    def num_active_arrows(self):
        num_arrows = 0
        for archer in self.archer_list:
            num_arrows += len(archer.weapons)
        return num_arrows


    @property
    def num_active_swords(self):
        num_swords = 0
        for knight in self.knight_list:
            num_swords += len(knight.weapons)
        return num_swords


    # Zombie Kills the Knight (also remove the sword)
    def zombit_hit_knight(self):
        for zombie in self.zombie_list:
            zombie_knight_list = pygame.sprite.spritecollide(
                zombie, self.knight_list, True
            )

            for knight in zombie_knight_list:
                knight.alive = False
                knight.weapons.empty()
                self.knight_list.remove(knight)
                if self.env_output:
                    print("Knight dead")


    # Zombie Kills the Archer
    def zombie_hit_archer(self):
        for zombie in self.zombie_list:
            zombie_archer_list = pygame.sprite.spritecollide(
                zombie, self.archer_list, True
            )

            for archer in zombie_archer_list:
                archer.alive = False
                archer.weapons.empty()
                self.archer_list.remove(archer)
                if self.env_output:
                    print("Archer dead")


    # Knight kills the Zombie
    def sword_hit(self):
        for knight in self.knight_list:
            for sword in knight.weapons:
                zombie_sword_list = pygame.sprite.spritecollide(
                    sword, self.zombie_list, True
                )

                for zombie in zombie_sword_list:
                    self.zombie_list.remove(zombie)
                    sword.knight.score += 1
                    if self.env_output:
                        print("Knight hit Zombie")


    # Archer kills the Zombie
    def arrow_hit(self):
        for archer in self.archer_list:
            for arrow in list(archer.weapons):
                zombie_arrow_list = pygame.sprite.spritecollide(
                    arrow, self.zombie_list, True
                )

                # For each zombie hit, remove the arrow, zombie and add to the score
                for zombie in zombie_arrow_list:
                    if self.env_output:
                        print("Archer hit Zombie")
                    archer.weapons.remove(arrow)
                    self.zombie_list.remove(zombie)
                    arrow.archer.score += 1


    # Zombie reaches the End of the Screen
    def zombie_endscreen(self, run, zombie_list):
        for zombie in zombie_list:
            if zombie.rect.y > const.SCREEN_HEIGHT - const.ZOMBIE_Y_SPEED:
                run = False
                if self.env_output:
                    print("Zombies reached end of screen")
        return run


    # Returns the current agent's observation (may be local or total scope)
    def observe(self, current_agent, full_obs=False):

        if not self.vector_state: # return the pixel image of game (not used due to time constraints)
            screen = pygame.surfarray.pixels3d(self.WINDOW)

            agent_obj = self.archer
            agent_position = (agent_obj.rect.x, agent_obj.rect.y)

            if not agent_obj.alive:
                cropped = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                min_x = agent_position[0] - 256
                max_x = agent_position[0] + 256
                min_y = agent_position[1] - 256
                max_y = agent_position[1] + 256
                lower_y_bound = max(min_y, 0)
                upper_y_bound = min(max_y, const.SCREEN_HEIGHT)
                lower_x_bound = max(min_x, 0)
                upper_x_bound = min(max_x, const.SCREEN_WIDTH)
                startx = lower_x_bound - min_x
                starty = lower_y_bound - min_y
                endx = 512 + upper_x_bound - max_x
                endy = 512 + upper_y_bound - max_y
                cropped = np.zeros_like(self.observation_space.low)
                cropped[startx:endx, starty:endy, :] = screen[
                    lower_x_bound:upper_x_bound, lower_y_bound:upper_y_bound, :
                ]
            return np.swapaxes(cropped, 1, 0)

        else:
            # get the agent
            agent = current_agent

            # get the agent position
            agent_state = agent.vector_state
            
            agent_pos = np.expand_dims(agent_state[0:2], axis=0)

            # get vector state of everything
            vector_state = self.get_vector_state(current_agent, full_obs)
            state = vector_state[:, -4:]
            is_dead = np.sum(np.abs(state), axis=1) == 0.0
            all_ids = vector_state[:, :-4]
            all_pos = state[:, 0:2]
            all_ang = state[:, 2:4]

            # get relative positions
            rel_pos = all_pos - agent_pos

            # get norm of relative distance
            norm_pos = np.linalg.norm(rel_pos, axis=1, keepdims=True) / np.sqrt(2)

            # kill dead things
            all_ids[is_dead] *= 0
            all_ang[is_dead] *= 0
            rel_pos[is_dead] *= 0
            norm_pos[is_dead] *= 0

            # combine the typemasks, positions and angles
            state = np.concatenate([all_ids, norm_pos, rel_pos, all_ang], axis=-1)

            # get the agent state as absolute vector
            # typemask is one longer to also include norm_pos
            if self.use_typemasks:
                typemask = np.zeros(self.typemask_width + 1)
                typemask[-2] = 1.0
            else:
                typemask = np.array([0.0])
            agent_state = agent.vector_state
            agent_state = np.concatenate([typemask, agent_state], axis=0)
            agent_state = np.expand_dims(agent_state, axis=0)

            # prepend agent state to the observation
            state = np.concatenate([agent_state, state], axis=0)

            if current_agent.is_archer: # If it is an archer

                if(full_obs):
                    # If we want a full observation of the game from the perspective of the archer
                    state = state.reshape([1, (self.num_tracked_full_obs + 1) * (self.vector_width + 1)])
                else:
                    # If we want the archer's local observation
                    state = state.reshape([1, (self.num_tracked + 1) * (self.vector_width + 1)])
            else: 
                # Must be a knight - using the observation dimensions that the knight policy was trained on
                state = state.reshape([1, (self.num_tracked_knight + 1) * (self.vector_width + 1)])

            return state


    # returns observation of the global environment -> PettingZoo comment (not used)
    def state(self, current_agent):
        """Returns an observation of the global environment."""
        if not self.vector_state:
            state = pygame.surfarray.pixels3d(self.WINDOW).copy()
            state = np.rot90(state, k=3)
            state = np.fliplr(state)
        else:
            state = self.get_vector_state(current_agent)

        return state


    # creates an observation of the environment as a (N x 5) matrix for each agent,
    # where N = num_archers + num_knights + num_swords + max_arrows + max_zombies
    def get_vector_state(self, arg_agent, full_obs):
        state = []
        typemask = np.array([])

        if(arg_agent.is_archer):
            if(full_obs):
                # Hacky way to get full game observation for KAZTrainingWrapper to use
                agents = [self.archer, self.knight] # will encode both players game information into the 'state' dictionary
            else:
                # Only want the archer information
                agents = [self.archer]
        else:
            # Only want the knight information
            agents = [self.knight]


        # per agent
        for _agent in agents:

            if _agent.alive:
                if self.use_typemasks:
                    typemask = np.zeros(self.typemask_width)
                    if _agent.is_archer:
                        typemask[1] = 1.0
                    elif _agent.is_knight:
                        typemask[2] = 1.0

                vector = np.concatenate((typemask, _agent.vector_state), axis=0)
                state.append(vector)
            else:
                if not self.transformer:
                    state.append(np.zeros(self.vector_width))
        
        for _agent in agents:

            if(_agent.is_knight):
                # handle swords
                if _agent.is_knight:
                    for sword in _agent.weapons:
                        if self.use_typemasks:
                            typemask = np.zeros(self.typemask_width)
                            typemask[4] = 1.0

                        vector = np.concatenate((typemask, sword.vector_state), axis=0)
                        state.append(vector)

                # handle empty swords
                if not self.transformer:
                    state.extend(
                        repeat(
                            np.zeros(self.vector_width),
                            1 - self.num_active_swords,
                        )
                    )
            else: # ARCHER
                # handle arrows
                if _agent.is_archer:
                    for arrow in _agent.weapons:
                        if self.use_typemasks:
                            typemask = np.zeros(self.typemask_width)
                            typemask[3] = 1.0

                        vector = np.concatenate((typemask, arrow.vector_state), axis=0)
                        state.append(vector)

                # handle empty arrows
                if not self.transformer:
                    state.extend(
                        repeat(
                            np.zeros(self.vector_width),
                            self.max_arrows - self.num_active_arrows,
                        )
                    )

        # handle zombies
        for zombie in self.zombie_list:
            if self.use_typemasks:
                typemask = np.zeros(self.typemask_width)
                typemask[0] = 1.0

            vector = np.concatenate((typemask, zombie.vector_state), axis=0)
            state.append(vector)


        # handle empty zombies
        if not self.transformer:
            state.extend(
                repeat(
                    np.zeros(self.vector_width),
                    self.max_zombies - len(self.zombie_list),
                )
            )

        return np.stack(state, axis=0)


    # Used to render game environment
    def enable_render(self):
        self.WINDOW = pygame.display.set_mode([const.SCREEN_WIDTH, const.SCREEN_HEIGHT])
        # self.WINDOW = pygame.Surface((const.SCREEN_WIDTH, const.SCREEN_HEIGHT))
        self.render_on = True
        self.draw()

    # Draws game environment
    def draw(self):
        self.WINDOW.fill((66, 40, 53))
        self.WINDOW.blit(self.left_wall, self.left_wall.get_rect())
        self.WINDOW.blit(self.right_wall, self.right_wall_rect)
        self.WINDOW.blit(self.floor_patch1, (500, 500))
        self.WINDOW.blit(self.floor_patch2, (900, 30))
        self.WINDOW.blit(self.floor_patch3, (150, 430))
        self.WINDOW.blit(self.floor_patch4, (300, 50))
        self.WINDOW.blit(self.floor_patch1, (1000, 250))

        # draw all the sprites
        self.zombie_list.draw(self.WINDOW)

        # Draw archer and knight components
        self.archer.weapons.draw(self.WINDOW)
        self.knight.weapons.draw(self.WINDOW)
        self.archer_list.draw(self.WINDOW)
        self.knight_list.draw(self.WINDOW)

    # Checks if the game has ended due to:
    # 1) One of the players dying
    # 2) A zombie has reached the other side of the game
    def check_game_end(self):
        # Zombie reaches the End of the Screen
        self.run = self.zombie_endscreen(self.run, self.zombie_list)

        if(self.run != False):
            # game only continues if both players are alive
            self.run = self.archer.alive and self.knight.alive


    # Reinitialises game variables; used in reset()
    def reinit(self):
        # Game Variables
        self.score = 0
        self.run = True
        self.zombie_spawn_rate = 0
        self.knight_player_num = self.archer_player_num = 0

        # Creating Sprite Groups
        self.zombie_list = pygame.sprite.Group()
        self.knight_list = pygame.sprite.Group()
        self.archer_list = pygame.sprite.Group()

        self.archer = Archer("Assistant_player")
        self.knight = Knight("Human_player")

        # Give both players a random starting position - offsets start from 0!
        knight_x_offset = self.np_random.integers(0, const.SCREEN_WIDTH)
        archer_x_offset = self.np_random.integers(0, const.SCREEN_WIDTH)

        self.archer.offset(archer_x_offset, 0)
        self.knight.offset(knight_x_offset, 0)
        
        self.knight_list.add(self.knight)
        self.archer_list.add(self.archer)

        self.draw()
        self.frames = 0