"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

from scipy import integrate

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

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

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, control_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.fric_coef = 0.0  # friction between floor and cart
        self.fric_rot = 0.0  # friction between cart and pole
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 45 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # temporary variables in order to implement digital pid (using velocity form)
        self.prev_mv = 0
        self.prev_err = 0
        self.dprev_err = 0

        # how long one response takes
        self.resp_time = 5

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode
        self.control_mode = control_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.stepstate = None
        self.state = []

        self.steps_beyond_terminated = None

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def step(self, action):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert -self.force_mag <= action and action <= self.force_mag, err_msg
        #assert self.stepstate is not None, "Call reset before using step method."

        self.reset()

        desired_state = np.array([0, 0, 0, 0])
        reward = 0.0

        for i in range(int(self.resp_time / self.tau)):
            
            x, x_dot, theta, theta_dot = self.stepstate
            # suppose that reference signal is 0 degree

            error = desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action)
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, action)

            def pend(y, t, F, m_c, m_p, l_p, g, k, b):
                x, x_dot, th, th_dot = y
                sintheta = np.sin(th)
                costheta = np.cos(th)
                temp = (
                    F + m_p * l_p * th_dot**2 * sintheta
                ) / (m_p + m_c)
                thetaacc = (g * sintheta - costheta * temp) / (
                    l_p * (4.0 / 3.0 - m_p * costheta**2 / (m_p + m_c))
                )
                xacc = temp - m_p * l_p * thetaacc * costheta / (m_p + m_c)

                ytdt = [x_dot, xacc, th_dot, thetaacc]
                return ytdt
            

            sol = integrate.odeint(pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))

            print(sol)

            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
            self.state.append(np.array(self.stepstate, dtype = np.float32))

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if self.render_mode == "human":
                self.render()

            # if terminated:
            #     if self.steps_beyond_terminated is None:
            #         self.steps_beyond_terminated = 0
            #         reward += 1.0
            #     else:
            #         if self.steps_beyond_terminated == 0:
            #             logger.warn(
            #                 "You are calling 'step()' even though this "
            #                 "environment has already returned terminated = True. You "
            #                 "should always call 'reset()' once you receive 'terminated = "
            #                 "True' -- any further steps are undefined behavior."
            #             )
            #         self.steps_beyond_terminated += 1
            #         reward += 0.0

            #     break
            # else:
            #     reward += 1.0

        
        
        return np.array(self.state), reward, {}
    
    def coefstep(self, action):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert -self.force_mag <= action and action <= self.force_mag, err_msg
        #assert self.stepstate is not None, "Call reset before using step method."

        score = 8.0

        for i in range(8):
            self.coefreset(i+1)

            desired_state = np.array([0, 0, 0, 0])

            for i in range(int(self.resp_time / self.tau)):
                
                x, x_dot, theta, theta_dot = self.stepstate
                # suppose that reference signal is 0 degree

                error = desired_state - self.stepstate

                if self.control_mode == 'pid1':
                    force = self.pidcontrol1(error, action)
                elif self.control_mode == 'pid2':
                    force = self.pidcontrol2(error, action)

                def pend(y, t, F, m_c, m_p, l_p, g, k, b):
                    x, x_dot, th, th_dot = y
                    sintheta = np.sin(th)
                    costheta = np.cos(th)
                    temp = (
                        F + m_p * l_p * th_dot**2 * sintheta
                    ) / (m_p + m_c)
                    thetaacc = (g * sintheta - costheta * temp) / (
                        l_p * (4.0 / 3.0 - m_p * costheta**2 / (m_p + m_c))
                    )
                    xacc = temp - m_p * l_p * thetaacc * costheta / (m_p + m_c)

                    ytdt = [x_dot, xacc, th_dot, thetaacc]
                    return ytdt
                

                sol = integrate.odeint(pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                    float(force), self.masscart, self.masspole, self.length, self.gravity
                ))

                self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
                self.state.append(np.array(self.stepstate, dtype = np.float32))

                terminated = bool(
                    sol[1][0] < -self.x_threshold
                    or sol[1][0] > self.x_threshold
                    or sol[1][2] < -self.theta_threshold_radians
                    or sol[1][2] > self.theta_threshold_radians
                )

                if self.render_mode == "human":
                    self.render()

                if terminated == True:
                    score -= 1.0
                    break

        
        
        return score, {}
    
    def linstep(self, action):
        '''
        [P, I, D] --> next_state, reward, truncated, {}
        '''

        self.reset()

        test_states = []
        desired_state = np.array([1, 0, 0, 0])
        reward = 0.0

        for i in range(500):
            
            x, x_dot, theta, theta_dot = self.stepstate
            # suppose that reference signal is 0 degree

            error = desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action)
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, action)

            def pend(y, t, F, m_c, m_p, l_p, g):
                x, x_dot, th, th_dot = y
                
                xacc = ((3 * g * m_p)/(4 - 3 * m_p)) * th + (4 / ((m_p + m_c) * (4 - 3 * m_p))) * F

                thetaacc = ((3 * g * (m_p + m_c)) / (l_p * (4 - 3 * m_p))) * th + (3 / (l_p * (4 - 3 * m_p))) * F

                ytdt = [x_dot, xacc, th_dot, thetaacc]
                return ytdt

            sol = integrate.odeint(pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity
            ))

            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
            self.state.append(np.array(self.stepstate, dtype = np.float32))

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if self.render_mode == "human":
                self.render()

            if terminated:
                if self.steps_beyond_terminated is None:
                    self.steps_beyond_terminated = 0
                    reward += 1.0
                else:
                    if self.steps_beyond_terminated == 0:
                        logger.warn(
                            "You are calling 'step()' even though this "
                            "environment has already returned terminated = True. You "
                            "should always call 'reset()' once you receive 'terminated = "
                            "True' -- any further steps are undefined behavior."
                        )
                    self.steps_beyond_terminated += 1
                    reward += 0.0

                break
            else:
                reward += 1.0

        return np.array(self.state), reward, {}
    
    def pidcontrol1(self, error, action):
        # action should be [K_p, K_i, K_d]
        error = error[2]
        mv = self.prev_mv + action[0] * (error - self.prev_err) + self.tau * action[1] * error + action[2] * (error - 2*self.prev_err + self.dprev_err) / self.tau

        if mv > self.force_mag:
            mv = self.force_mag
        
        if mv < -self.force_mag:
            mv = -self.force_mag

        self.prev_mv = mv
        self.dprev_err = self.prev_err
        self.prev_err = error
        
        return mv
    
    def pidcontrol2(self, error, action):
        # action should be [[K_p_c, K_p_th], [K_i_c, K_i_th], [K_d_c, K_d_th]]
        error = [error[0], error[2]]
        
        if self.prev_err == 0:
            self.prev_err = [0, 0]
            self.dprev_err = [0, 0]

        for i in range(3):
            action[i][0] = -action[i][0]

        print(error, action[0])
        
        mv = self.prev_mv + (np.dot(np.array(action[0]), np.array(error) - np.array(self.prev_err)) 
                             + self.tau * np.dot(np.array(action[1]), np.array(error)) 
                             + np.dot(np.array(action[2]), np.array(error) - 2*np.array(self.prev_err) + np.array(self.dprev_err)) / self.tau)

        if mv > self.force_mag:
            mv = self.force_mag
        
        if mv < -self.force_mag:
            mv = -self.force_mag

        self.prev_mv = mv
        self.dprev_err = self.prev_err
        self.prev_err = error


        print(mv)
        for i in range(3):
            action[i][0] = -action[i][0]
        return mv

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.stepstate = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None

        self.integral = 0
        self.prev_err = 0

        self.state = []
        self.state.append(self.stepstate)

        if self.render_mode == "human":
            self.render()
        return np.array(self.stepstate, dtype=np.float32), {}
    
    def coefreset(
        self,
        divide,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        self.stepstate = [0,0, 0.2 * (4.5 - divide) / 3.5,0]
        self.steps_beyond_terminated = None

        self.integral = 0
        self.prev_err = 0

        self.state = []
        self.state.append(self.stepstate)

        if self.render_mode == "human":
            self.render()
        return np.array(self.stepstate, dtype=np.float32), {}

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

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.stepstate is None:
            return None

        x = self.stepstate

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

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