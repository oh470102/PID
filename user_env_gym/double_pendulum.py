"""
File Name : double_pendulum.py

"""


import math
import scipy.stats
from scipy import integrate
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class DoublePendEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{-20, 20}` for default.
    Action value corresponds to the force pushing or pulling cart
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Amount of force        |

    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards
    There are various modes of rewards implemented, and you can choose one of them when initialize.
    1) Constant reward (default) (reward_mode = 0)
        Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
        including the termination step, is allotted. The threshold for rewards is 475 for v1.
    2) Discrete reward (reward_mode = 1)
        Gives reward = 2 when position is between -1 and 1. Else, gives reward = 1
    3) Continuous reward (reward_mode = 2)
        Gives reward according to normal distribution. (Time consuming)

    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 100,
    }

    def __init__(self, render_mode: Optional[str] = None, reward_mode: int = 0):
        # assert reward_mode == 0 or reward_mode == 1 or reward_mode == 2, "Wrong reward mode parameter" 

        self.gravity = 9.8
        self.masspend1 = 0.5
        self.masspend2 = 0.5
        self.lengthpend1 = 1.5
        self.lengthpend2 = 1.5 
        self.torque_mag = 20.0  # 힘 값 수정 필요
        self.tau = 0.01  # seconds between state updates

        # consider rotational friction
        self.fric_coef1 = 0.05
        self.fric_coef2 = 0.05

        self.theta1_threshold_radian = 1.0
        self.theta2_threshold_radian = 0.6

        self.prev_state = None

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                np.finfo(np.float32).max,  # theta1
                np.finfo(np.float32).max,  # theta1dot
                np.finfo(np.float32).max,  # theta2
                np.finfo(np.float32).max,  # theta2dot
            ],
            dtype=np.float32,
        )

        # Define action and observation space. spaces.Discrete() for discrete action/observation space
        self.action_space = spaces.Box(-self.torque_mag, self.torque_mag, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.reward_mode = reward_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.episode_length = 0

        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = "invalid action detected. Please set action value between {0} and {1}".format(-self.torque_mag, self.torque_mag)
        # assert -self.torque_mag <= action and action <= self.torque_mag, err_msg
        assert self.state is not None, "Call reset before using step method."
        theta1, theta1_dot, theta2, theta2_dot = self.state
        torque = action

        def pend(y, t, tau, m_1, m_2, l_1, l_2, g, b_1 = 0, b_2 = 0):
            th1, th1_dot, th2, th2_dot = y
            sinth1 = np.sin(th1)
            sinth2 = np.sin(th2)
            sindelth = np.sin(th1 - th2)
            cosdelth = np.cos(th1 - th2)

            tmp1_1 = (3 * l_1)/(2 * l_2) * th1_dot * th2_dot * sindelth
            tmp1_2 = -g * (3 * sinth2)/(2 * l_2)
            tmp1_3 = (3 * l_1)/(2 * l_2) * th1_dot * sindelth * (th1_dot - th2_dot)
            tmp1 = tmp1_1 + tmp1_2 + tmp1_3 + 3 * b_2 * (th1_dot - th2_dot) / l_2

            tmp2_1 = -tau + 0.5 * m_2 * l_1 * l_2 * th1_dot * th2_dot * sindelth
            tmp2_2 = 0.5 * m_1 * g * l_1 * sinth1 + m_2 * g * l_1 * sinth1
            tmp2_3 = -0.5 * m_2 * l_1 * l_2 * th2_dot * sindelth * (th1_dot - th2_dot)
            tmp2 = tmp2_1 + tmp2_2 + tmp2_3 + b_1 * th1_dot
            
            tmp3 = 0.75 * m_2 * l_1**2 * cosdelth**2 - (m_2 + (m_1 / 3)) * l_1**2

            th1_2dot = (tmp2 + 0.5 * m_2 * l_1 * l_2 * tmp1 * cosdelth) / tmp3
            th2_2dot = tmp1 - (3 * l_1 * cosdelth * (tmp2 + 0.5 * m_2 * l_1 * l_2 * tmp1 * cosdelth)) / (2 * l_2 * tmp3)

            ytdt = [th1_dot, th1_2dot, th2_dot, th2_2dot]
            return ytdt

        sol = integrate.odeint(pend, [theta1, theta1_dot, theta2, theta2_dot], [0, self.tau], args = (
            torque, self.masspend1, self.masspend2, self.lengthpend1, self.lengthpend2, self.gravity, self.fric_coef1, self.fric_coef2
        ))


        self.state = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])

        terminated = bool(
            theta1 < np.pi - self.theta1_threshold_radian
            or theta1 > np.pi + self.theta1_threshold_radian 
            or theta2 < np.pi - self.theta2_threshold_radian 
            or theta2 > np.pi + self.theta2_threshold_radian 
        )

        reward = 0.0

        if not terminated:
            
            if self.reward_mode == 0:
                reward = -1
            if self.reward_mode == 1:
                norm_dist_th1 = scipy.stats.norm(loc = 0, scale = 0.3)
                reward = 0.3 * math.sqrt(2 * np.pi) * norm_dist_th1.pdf(np.pi - self.state[0]) - 0.5
                # if np.pi - self.state[0] <= 0.3 and self.state[0] <= np.pi + 0.3:
                #     reward = 2.0
                # else:
                #     reward = 1.0
                
                if self.state[1] > 10 or self.state[3] > 10:
                    reward -= 1.0

            if self.reward_mode == 2:
                norm_dist_x = scipy.stats.norm(loc = 0, scale = 0.5)
                norm_dist_theta = scipy.stats.norm(loc = 0, scale = 0.1)
                reward = 0.5 * math.sqrt(2 * 3.14) * norm_dist_x.pdf(self.state[0]) + 0.1 * math.sqrt(2 * 3.14) * norm_dist_theta.pdf(self.state[2])
            if self.reward_mode == 3:
                reward = -((np.pi - self.state[0])**2) - 0.1 * (np.pi - self.state[2])**2 - 0.001 * torque**2
                if self.prev_state != None:
                    reward -= 0.5 * ((self.state[0] - self.prev_state[0]) / self.tau)**2 + 0.05 * ((self.state[2] - self.prev_state[2]) / self.tau)**2

                if isinstance(reward, float) is False: reward = reward.squeeze()
                
            if self.reward_mode == 4:
                in_bound = bool(np.radians(135) < theta1 < np.radians(135+90) and np.radians(135) < theta2 < np.radians(135+90))
                if in_bound: reward = 1
                else: reward = -1

        if terminated:
            reward = 0

        # elif self.steps_beyond_terminated is None:
        #     # Pole just fell!
        #     self.steps_beyond_terminated = 0
        #     reward = 1.0
        # else:
        #     if self.steps_beyond_terminated == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned terminated = True. You "
        #             "should always call 'reset()' once you receive 'terminated = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_terminated += 1
        #     reward = 0.0

        self.prev_state = self.state
        self.episode_length += 1

        self.outputstate = [np.cos(self.state[0]), np.sin(self.state[0]), np.cos(self.state[2]), np.sin(self.state[2]), self.state[1], self.state[3]]

        if self.render_mode == "human":
            self.render()
        return np.array(self.outputstate, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        
        self.state = self.np_random.uniform(low=[np.pi-0.1, 0., np.pi-0.1, 0.], high=[np.pi+0.1, 0., np.pi+0.1, 0.0], size=(4,))
        self.steps_beyond_terminated = None
        self.prev_state = None
        self.episode_length = 0

        self.outputstate = [np.cos(self.state[0]), np.sin(self.state[0]), np.cos(self.state[2]), np.sin(self.state[2]), self.state[1], self.state[3]]
        if self.render_mode == "human":
            self.render()

        return np.array(self.outputstate, dtype=np.float32), {}


    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = 5 * 2
        scale = self.screen_width / world_width
        polewidth = 8.0
        polelen1 = scale * (self.lengthpend1)
        polelen2 = scale * (self.lengthpend2)

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen1,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[0] + np.pi) # 여기 좌표 체계가 잘 이해되지 않는다..
            coord = (coord[0] + self.screen_width/2, coord[1] + self.screen_height/2)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[2] + np.pi)
            coord = (coord[0] + self.screen_width/2 + polelen1 * np.sin(x[0]), coord[1] + self.screen_height/2 - polelen1 * np.cos(x[0]))
            
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))


        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
