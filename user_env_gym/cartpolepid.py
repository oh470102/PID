"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np
import time
from scipy import integrate
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import user_env_gym.controlutil as ctut

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
        self.resp_time = 100

        # start matlab engine
        self.eng = ctut.start_matengine()

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

        self.PID = np.zeros(3)
        self.PID_MIMO = np.zeros(6)
        self.PID_MIMO_last = np.zeros(6)
        self.PID_MIMO_BASELINE = np.array([-46, 130, -25, 126, -20,  16])
        self.time = 0

        self.best_stability = 0
        self.best_PID = None
        self.best_ISE = 737              # baseline ISE is 737.
        self.stability_threshold = 0.4
        self.lin_stability_threshold = 0 # fix to np.min(1, -ctut.lin_stability_MIMO(self.PID_MIMO_BASELINE)/2) later.

        self.x_trajectory = []
        self.theta_trajectory = []

        self.A = [[0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.7945946, 0.0, 0.0],
                  [0.0, 17.4810811, 0.0, 0.0]]
        self.B = [[0], [0], [0.982801], [1.62162]]
        self.C = [0, 1, 0, 0]
        self.D = [0]
        self.E = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
        
        self.Am = [[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0.7945946, 0, 0],
              [0, 17.4810811, 0, 0]]
        self.Bm = [[0],
             [0],
             [0.982801],
             [1.62162]]
        self.Cm = [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        self.Dm = [[0],
              [0]]
        self.Em = [[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

        self.prev_stability = None
        self.desired_state = np.array([1,0,0,0])

        self.x_upper_bound, self.x_lower_bound, self.theta_upper_bound, self.theta_lower_bound, self.prev_ISE = self.get_bound_by_rollout()
        print(self.x_upper_bound, self.x_lower_bound, self.theta_upper_bound, self.theta_lower_bound, self.prev_ISE)

    def get_bound_by_rollout(self, generosity=0.35):
        '''
        calculates upper and lower bounds of state variables through a rollout of baseline PID
        larger generosity means the bound gets more generous (greater bound)
        also returns baseline ISE for future reward calculation
        '''

        self.iterreset(custom=np.array([0,0,0,0]))
        rollout_traj_x, rollout_traj_theta, full_traj = [], [], []
        action = self.PID_MIMO_BASELINE
        reached_setpoint_at = [None, None] # (x, theta)
 
        for i in range(500):
            
            x, x_dot, theta, theta_dot = self.stepstate
            rollout_traj_x.append(x); rollout_traj_theta.append(theta); full_traj.append(np.array([x, theta]))
            
            error = self.desired_state - self.stepstate
            if reached_setpoint_at[0] is not None and abs(error[0]) < 0.05: reached_setpoint_at[0] = i
            if reached_setpoint_at[1] is not None and abs(error[2]) < 0.05: reached_setpoint_at[1] = i

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action) 
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, self.group_MIMO(self.PID_MIMO_BASELINE)) 

            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))

            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
            self.state.append(np.array(self.stepstate, dtype = np.float32))

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if terminated: break     
        
        rollout_traj_x = np.array(rollout_traj_x[reached_setpoint_at[0]: ])
        max_x = np.max(rollout_traj_x)
        x_upper_bound = max_x + generosity * (max_x - self.desired_state[0])
        x_lower_bound = self.desired_state[0] - (x_upper_bound - self.desired_state[0])

        rollout_traj_theta = np.array(rollout_traj_theta[reached_setpoint_at[1]: ])
        max_theta = np.max(rollout_traj_theta)
        theta_upper_bound = max_theta + generosity * (max_theta - self.desired_state[2])
        theta_lower_bound = self.desired_state[2] - (theta_upper_bound - self.desired_state[2])

        return x_upper_bound, x_lower_bound, theta_upper_bound, theta_lower_bound, ctut.calISE(full_traj, np.array([self.desired_state[0], self.desired_state[2]]))

    def get_curr_stability(self, MIMO=False):
        '''
        returns stability using the current PID (SISO or MIMO, stored as class instance var.)
        '''

        if not MIMO:
            P, I, D = tuple(map(int, list(self.PID)))
            stability = ctut.lin_stability_SISO(self.eng, P, I, D, self.A, self.B, self.C, self.D, self.E)

            return -stability
        
        if MIMO:
            P1, P2, I1, I2, D1, D2 = tuple(map(int, list(self.PID_MIMO)))
            stability = ctut.lin_stability_MIMO(self.eng, [P1, P2],[I1, I2],[D1,D2], self.Am, self.Bm, self.Cm, self.Dm, self.Em)

            return -stability

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def pend(self, y, t, F, m_c, m_p, l_p, g, k, b):
        x, x_dot, th, th_dot = y
        sintheta = np.sin(th)
        costheta = np.cos(th)
        temp = (
            F - m_p * l_p * th_dot**2 * sintheta
        ) / (m_p + m_c)
        thetaacc = (g * sintheta + costheta * temp) / (
            l_p * (4.0 / 3.0 - m_p * costheta**2 / (m_p + m_c))
        )
        xacc = temp + m_p * l_p * thetaacc * costheta / (m_p + m_c)

        ytdt = [x_dot, xacc, th_dot, thetaacc]
        return ytdt
    
    def restore_setup(self):

        '''
        Restores the original cart-pole setup 
        when the controller becomes too unstable.
        Baseline PID is used for this process.
        '''

        desired_state = np.array([0,0,0,0])
        action = None
        initial_condition = self.stepstate.copy()

        self.prev_mv=0
        self.prev_err = 0
        self.dprev_err = 0

        # print("Restoration initiated...", end='')

        while True:

            x, x_dot, theta, theta_dot = tuple(self.stepstate)
            error = desired_state - self.stepstate

            if abs(error[0]) < 0.025 and abs(error[2]) < 0.025: 
                # print("succeeded!")
                # print(sols)
                break

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action) 
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, self.group_MIMO(self.PID_MIMO_BASELINE)) 

            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))

            self.stepstate = np.array([sol[1][0], sol[1][1], sol[1][2], sol[1][3]])

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if terminated: 
                print("FAILED!")
                print(tuple(initial_condition))
                # raise Exception

            if self.render_mode == "human":
                self.render() 

    def step_online(self, action):
        '''
        NOTE: iterreset() must be ran internally the very first time only for future training (real-like)
        assumes MIMO by default
        '''

        # reset env 
        self.iterreset(custom=np.array([0,0,0,0])) 
        # self.iterreset()

        # save current PID 
        self.PID_MIMO_last = self.PID_MIMO.copy()

        # update & clip PID by action
        self.PID_MIMO += action
        self.PID_MIMO = self.clip_PID_MIMO(self.PID_MIMO)

        # record x, theta trajectory to keep track of controller stability
        x_reached_setpoint, theta_reached_setpoint = False, False
        x_list, theta_list, trajectory = [], [], []
        reward = 0

        for i in range(500):
            
            # calculate linear stability, just once prior to simulation
            if i == 0:
                lin_stability = self.get_curr_stability(MIMO=True)
                # print(f"stability: {lin_stability}")

                # if too unstable, break immediately.
                if lin_stability < self.lin_stability_threshold:
                    
                    # print("TOO UNSTABLE!")

                    # re-set PID to previous PID value
                    self.PID_MIMO = self.PID_MIMO_last.copy()

                    break

            # record trajectory
            x, x_dot, theta, theta_dot = self.stepstate
            x_list.append(x); theta_list.append(theta); trajectory.append(self.stepstate)

            # check whether x, theta is near setpoint
            if x_reached_setpoint is False and abs(x - self.desired_state[0]) < 0.05: x_reached_setpoint = True
            if theta_reached_setpoint is False and abs(theta - self.desired_state[2]) < 0.05: theta_reached_setpoint = True

            # if object's position gets out-of-bound, break immediately.
            if x_reached_setpoint and not self.x_lower_bound < x < self.x_upper_bound:

                # print(f"{x:.3f} was out of bound (position)")

                # re-set PID to previous PID value
                self.PID_MIMO = self.PID_MIMO_last.copy()

                # restore original setup
                self.restore_setup()

                break
            
            # if object's angle gets out-of-bound, break immediately.
            if theta_reached_setpoint and not self.theta_lower_bound < theta < self.theta_upper_bound:

                # print(f"{theta:.3f} was out of bound (angle).")

                # re-set PID to previous PID value
                self.PID_MIMO = self.PID_MIMO_last.copy()

                # restore original setup
                self.restore_setup()

                break

            error = self.desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action) 
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, self.group_MIMO(self.PID_MIMO)) 

            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))

            self.stepstate = np.array([sol[1][0], sol[1][1], sol[1][2], sol[1][3]])
            self.state.append(np.array(self.stepstate, dtype = np.float32))

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if terminated: break

            if self.render_mode == "human":
                self.render()        

        # set next_state as updated PID 
        next_state = self.PID_MIMO.copy()

        # controller was too unstable, and action was not [0] x 6
        if list(self.PID_MIMO_last) == list(self.PID_MIMO) and list(action) != list(np.zeros(6, dtype=np.int64)):
            reward = -1
        
        # controller was stable, then calculate ISE improvements
        else:
            curr_ISE = ctut.calISE(trajectory, self.desired_state)

            # print if new best ISE is found 
            if curr_ISE < self.best_ISE: 
                self.best_ISE = curr_ISE.copy() 
                print(f"best ISE: {self.best_ISE:.2f} by {self.PID_MIMO}")
            
            reward = (self.prev_ISE - curr_ISE) / 10
            self.prev_ISE = curr_ISE

        # episode ends after 50 PID updates
        truncated = True if self.time >= 50 else False
        self.time += 1       

        return next_state, reward, False, truncated, {}

    def step(self, action):

        self.iterreset()

        desired_state = np.array([1, 0, 0, 0])
        reward = 0.0
        x_list, theta_list = [], []

        for i in range(1000):
            
            x, x_dot, theta, theta_dot = self.stepstate
            x_list.append(x); theta_list.append(theta)
            # suppose that reference signal is 0 degree

            error = desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, action) 
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, action) #+ 3 * np.random.randn(1)[0]

            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))

            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
            self.state.append(np.array(self.stepstate, dtype = np.float32))

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if terminated: break
            else: reward += 1

            if self.render_mode == "human":
                self.render()        
        
        return reward, x_list, theta_list
    
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
        action = [dP, dI, dD] <numpy array>
        '''
        self.prev_stability = self.get_curr_stability()
        self.PID += action
        self.PID = np.clip(self.PID, 10, 150)
        self.time += 1

        desired_state = np.array([0, 0, 0, 0])
        score = 0.0

        self.iterreset()
        for i in range(500):

            x, x_dot, theta, theta_dot = self.stepstate
            # suppose that reference signal is 0 degree

            error = desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, self.PID)
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, self.PID)

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
                    score += 0.0
                else:
                    if self.steps_beyond_terminated == 0:
                        logger.warn(
                            "You are calling 'step()' even though this "
                            "environment has already returned terminated = True. You "
                            "should always call 'reset()' once you receive 'terminated = "
                            "True' -- any further steps are undefined behavior."
                        )
                    self.steps_beyond_terminated += 1
                    score += 0.0

                break
            else:
                score += 1.0
        
        new_stability = self.get_curr_stability()
        reward = 50 * (new_stability - self.prev_stability) 
        self.prev_stability = new_stability

        truncated = True if self.time >= 100 else False 

        return self.PID, reward, False, truncated, {}
    
    def linstep_MIMO(self, action):

        '''
        action = [dP1, dP2, dI1, dI2, dD1, dD2] <numpy array>
        '''
        self.prev_stability = self.get_curr_stability(MIMO=True)
        self.PID_MIMO += action
        self.PID_MIMO = self.clip_PID_MIMO(self.PID_MIMO)
        self.time += 1

        desired_state = np.array([1, 0, 0, 0])
        score = 0.0

        #self.iterreset()

        #''' NOTE: temporarily commented to increase training speed.
        x_list, theta_list = [], []
        for i in range(1000):

            x, x_dot, theta, theta_dot = self.stepstate
            x_list.append(x); theta_list.append(theta)
            # suppose that reference signal is 0 degree

            error = desired_state - self.stepstate

            if self.control_mode == 'pid1':
                force = self.pidcontrol1(error, self.PID)
            elif self.control_mode == 'pid2':
                force = self.pidcontrol2(error, self.group_MIMO(self.PID_MIMO))

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
                    score += 0.0
                else:
                    if self.steps_beyond_terminated == 0:
                        logger.warn(
                            "You are calling 'step()' even though this "
                            "environment has already returned terminated = True. You "
                            "should always call 'reset()' once you receive 'terminated = "
                            "True' -- any further steps are undefined behavior."
                        )
                    self.steps_beyond_terminated += 1
                    score += 0.0

                break
            else:
                score += 1.0
        #'''
        
        new_stability = self.get_curr_stability(MIMO=True)
        reward = 5 * (new_stability - self.prev_stability) 
        self.prev_stability = new_stability

        # save good stability & PIDs
        if self.prev_stability > self.best_stability:
            self.best_stability = self.prev_stability
            self.best_PID = self.PID_MIMO

            # print current best PID
            print(f"best stability: {self.best_stability:.2f} by {self.best_PID}")

        truncated = True if self.time >= 75 else False 

        return self.PID_MIMO, reward, False, truncated, [x_list, theta_list]
    
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


       # print(error, action[0])
        
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

        return mv

    def combine_MIMO(self, pos_PID, angle_PID):
        '''
        transforms [P1, I1, D1] & [P2, I2, D2] to [P1, P2, I1, I2, D1, D2] as <numpy.array>
        '''
        combined_PID = np.zeros(6)

        for i in range(0, 6, 2):
            combined_PID[i] = pos_PID[i//2]
            combined_PID[i+1] = angle_PID[i//2] 

        return combined_PID

    def divide_MIMO(self, combined_PID):
        '''
        transforms [P1, P2, I1, I2, D1, D2] to [P1, I1, D1] & [P2, I2, D2]
        '''
        
        pos_PID, angle_PID = np.zeros(3), np.zeros(3)

        for i in range(0, 6):
            if i % 2 == 0:
                pos_PID[i//2] = combined_PID[i]
            else:
                angle_PID[i//2] = combined_PID[i]

        return pos_PID, angle_PID
    
    def group_MIMO(self, combined_PID):
        '''
        transforms [P1, P2, I1, I2, D1, D2] to [[P1, P2], [I1, I2], [D1, D2]]
        '''
        
        ret = []

        for i in range(0, 6, 2):
            ret.append([combined_PID[i], combined_PID[i+1]])

        return np.array(ret)

    def get_random_MIMO(self):
        '''
        returns randomly initialized np.array([p1, p2, i1, i2, d1, d2])
        '''
        
        position_PID = np.random.randint(low=5, high=150, size=3) * -1
        angle_PID = np.random.randint(low=5, high=150, size=3) 

        return self.combine_MIMO(position_PID, angle_PID)

    def clip_PID_MIMO(self, PID_MIMO):
        '''
        clips [P1, P2, I1, I2, D1, D2] within (10, 150) 
        sign is considered.
        '''

        P1, P2, I1, I2, D1, D2 = tuple(map(int, list(PID_MIMO)))
        
        clipped_pos_PID = np.clip(np.array([P1, I1, D1]), -150, -5)
        clipped_angle_PID = np.clip(np.array([P2, I2, D2]), 5, 150)

        return self.combine_MIMO(clipped_pos_PID, clipped_angle_PID)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        custom_PID = None,
        custom_PID_MIMO = None,
        MIMO=False,
        online=False
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.

        if online is False:
            self.PID = np.random.randint(low=10, high=150, size=3) if custom_PID is None else custom_PID
            self.PID_MIMO = self.get_random_MIMO() if custom_PID_MIMO is None else custom_PID_MIMO

        elif online is True:
            # give some noise to baseline PID for better exploration chances
            self.PID_MIMO = self.PID_MIMO_BASELINE.copy() + np.random.uniform(low=-5, high=5, size=6)

        self.time = 0
        self.prev_stability = None

        if self.render_mode == "human":
            self.render()

        if MIMO is False: return (self.PID.copy(), {})
        elif MIMO is True: return (self.PID_MIMO.copy(), {})
        # returns a copy to prevent unseen side-effects.

    def iterreset(self, seed=None, options=None, custom=None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.stepstate = self.np_random.uniform(low=low, high=high, size=(4,)) if custom is None else custom
        self.steps_beyond_terminated = None

        self.integral = 0
        self.prev_err = 0
        self.prev_mv = 0

        self.state = []
        self.state.append(self.stepstate)
  
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