"""
The classic Cart-Pole environment, with slight tweaks.
See comment under class definition to see major changes.

Reference:
    - ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"] (https://ieeexplore.ieee.org/document/6313077)
    - code from: http://incompleteideas.net/sutton/book/code/pole.c
    - permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Optional, Union
import numpy as np
from scipy import integrate
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import user_env_gym.controlutil as ctut

class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    """
    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values from the interval [-10, 10] which indicates the 
    amount of force (N) to push the cart with.

    | Sign | Action (Force)         |
    |------|------------------------|
    | +    | Push cart to the right |
    | -    | Push cart to the left  |

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

    def __init__(self, render_mode: Optional[str] = None, control_mode: Optional[str] = None) -> None:

        # Physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  
        self.polemass_length = self.masspole * self.length
        self.fric_coef = 0.0                # friction between floor and cart
        self.fric_rot = 0.0                 # friction between cart and pole
        self.force_mag = 10.0               # boundary for force magnitude
        self.tau = 0.02                     # seconds between state updates
        self.kinematics_integrator = "euler"

        # MATLAB engine (for stability calculation)
        self.eng = ctut.start_matengine()

        # Angle, position thresholds
        self.theta_threshold_radians = 45 * math.pi / 180
        self.x_threshold = 2.4
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

        # Pygame stuff
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.stepstate = None
        self.render_mode = render_mode
        self.steps_beyond_terminated = None

        # Digital PID (velocity form) variables
        self.prev_mv = 0
        self.prev_err = 0
        self.dprev_err = 0
        self.control_mode = control_mode   
        self.desired_state = np.array([1,0,0,0])

        # More PID stuff (for reward calculation)
        self.PID = None
        self.PID_last = None
        self.PID_SISO_BASELINE = np.array([72, 136, 10])
        self.PID_MIMO_BASELINE = np.array([-46, 130, -25, 126, -20,  16])

        # More PID stuff 2 (for saving best-performers)
        self.best_stability = None
        self.prev_stability = None
        self.best_PID = None
        self.best_ISE = None         

        # for stability-wise guarantees
        self.lin_stability_threshold = 0.4 
        self.x_upper_bound, self.x_lower_bound, self.theta_upper_bound, self.theta_lower_bound, self.prev_ISE = self.get_bound_by_rollout()
        self.best_ISE = self.prev_ISE

        # Miscellaneous
        self.time = 0

        # Matrices for stability calculation
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

    def get_bound_by_rollout(self, generosity: float = 0.35) -> tuple:

        '''
        calculates upper and lower bounds of state variables (position, angle)
        through a rollout (simulation) of baseline PID

            - larger generosity means larger bound
            - also returns ISE of baseline PID for future reward calculation
        '''

        # reset env (may change to random)
        self.iterreset(custom=np.array([0,0,0,0]))

        # setup variables
        rollout_traj_x, rollout_traj_theta, full_traj = [], [], []
        reached_setpoint_at = [None, None]
 
        # use baseline PID
        PID = self.PID_MIMO_BASELINE if self.control_mode == 'pid2' else self.PID_SISO_BASELINE

        # env loop
        # NOTE: we know baseline PID will not fail, so environment termination condition was not considered.
        for i in range(500):
            
            # record current state variables (x, v, theta, w)
            x, x_dot, theta, theta_dot = self.stepstate
            rollout_traj_x.append(x); rollout_traj_theta.append(theta); full_traj.append(np.array([x, theta]))
            
            # calculate error, force
            error = self.desired_state - self.stepstate
            force = self.pidcontrol(error, use_diff_PID=PID)

            # record the instance at which setpoint was reached
            if reached_setpoint_at[0] is not None and abs(error[0]) < 0.05: reached_setpoint_at[0] = i
            if reached_setpoint_at[1] is not None and abs(error[2]) < 0.05: reached_setpoint_at[1] = i

            # calculate new state variables 
            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))
            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])
        
        # calculate upper, lower bounds for position
        rollout_traj_x = np.array(rollout_traj_x[reached_setpoint_at[0]: ])
        max_x = np.max(rollout_traj_x)
        x_upper_bound = max_x + generosity * (max_x - self.desired_state[0])
        x_lower_bound = self.desired_state[0] - (x_upper_bound - self.desired_state[0])

        # calculate upper, lower bounds for angle
        rollout_traj_theta = np.array(rollout_traj_theta[reached_setpoint_at[1]: ])
        max_theta = np.max(rollout_traj_theta)
        theta_upper_bound = max_theta + generosity * (max_theta - self.desired_state[2])
        theta_lower_bound = self.desired_state[2] - (theta_upper_bound - self.desired_state[2])

        return x_upper_bound, x_lower_bound, theta_upper_bound, theta_lower_bound, ctut.calISE(full_traj, np.array([self.desired_state[0], self.desired_state[2]]))

    def get_curr_stability(self) -> float:

        '''
        returns the current PID coefficients' stability
            - the sign is reversed, so the more positive, the more stable
            - NOTE: stability calculation depends on the control_mode
        '''

        MIMO = True if self.control_mode == 'pid2' else False

        if not MIMO:
            P, I, D = tuple(map(float, list(self.PID)))
            stability = ctut.lin_stability_SISO(self.eng, P, I, D, self.A, self.B, self.C, self.D, self.E)

            return -stability
        
        if MIMO:
            P1, P2, I1, I2, D1, D2 = tuple(map(float, list(self.PID)))
            stability = ctut.lin_stability_MIMO(self.eng, [P1, P2], [I1, I2], [D1,D2], self.Am, self.Bm, self.Cm, self.Dm, self.Em)

            return -stability

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def pend(self, y, t, F, m_c, m_p, l_p, g, k, b) -> list:

        '''
        real-dynamics calculation for cartpole
        '''

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
    
    def restore_setup(self) -> None:

        '''
        Restores the original cart-pole setup 
        when the controller becomes too unstable.
        Baseline PID is used for this process.
        '''

        # desired_state is now the initial condition (around [0,0,0,0])
        desired_state = np.array([0,0,0,0])

        # record initial condition for logging purposes
        initial_condition = self.stepstate.copy()

        # reset PID variables 
        # NOTE: do NOT call iterreset() since that automatically bring the cart to the initial condition
        self.prev_mv = 0
        self.prev_err = 0
        self.dprev_err = 0

        # use baseline PID
        PID = self.PID_MIMO_BASELINE if self.control_mode == 'pid2' else self.PID_SISO_BASELINE

        # print("Restoration initiated...", end='')

        # loop until initial condition restored
        while True:

            # retrieve current env state variables
            x, x_dot, theta, theta_dot = tuple(self.stepstate)

            # calculate error, force
            error = desired_state - self.stepstate
            force = self.pidcontrol(error, use_diff_PID=PID)

            # break if error is sufficiently small for both position & angle (initial state is restored
            if abs(error[0]) < 0.025 and abs(error[2]) < 0.025: 
                break

            # otherwise, calculate new env stepstates
            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))
            self.stepstate = np.array([sol[1][0], sol[1][1], sol[1][2], sol[1][3]])

            # check if env terminated
            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            # this means the restoration process failed
            if terminated: 
                print("FATAL: Restoration process failed. ")

                # TODO: implement what to do if restoration fails
                break

            if self.render_mode == "human":
                self.render() 

    def step_online(self, action: np.ndarray) -> tuple

        '''
        For online-tuning with real dynamics.
            - follows a very similar structure to gym's step() function

            - receives action and returns (s', r, terminated, truncated, info)
                - next_state (s') is simply current_state + action
                - reward is (previous_ISE - current_ISE) / 10 (thus, the improvement in ISE scaled by 1/10).
                - terminated is always False
                - truncated is True after 50 time-steps
                - info is a list containing position, angle trajectory

            - action must be a np.ndarray of appropriate size (agreeing with control_mode)
        '''

        # action size must agree with control mode
        assert action.size == self.PID.size, f'action size ({action.size}) does not agree with PID size ({self.PID.size})'

        # reset env 
        self.iterreset(custom=np.array([0,0,0,0])) 

        # save current PID 
        self.PID_last = self.PID.copy()

        # update & clip PID by action
        self.PID += action
        self.PID = self.clip_PID()

        # record x, theta trajectory to keep track of controller stability
        x_reached_setpoint, theta_reached_setpoint = False, False
        x_list, theta_list, trajectory = [], [], []
        reward = 0

        for i in range(500):
            
            # calculate linear stability, just once prior to simulation
            if i == 0:
                lin_stability = self.get_curr_stability()

                # if too unstable, break immediately.
                if lin_stability < self.lin_stability_threshold:
                
                    # re-set PID to previous PID value
                    self.PID = self.PID_last.copy()

                    break

            # record trajectory
            x, x_dot, theta, theta_dot = self.stepstate
            x_list.append(x); theta_list.append(theta); trajectory.append(self.stepstate)

            # check whether x, theta is near setpoint
            if x_reached_setpoint is False and abs(x - self.desired_state[0]) < 0.05: x_reached_setpoint = True
            if theta_reached_setpoint is False and abs(theta - self.desired_state[2]) < 0.05: theta_reached_setpoint = True

            # if object reached setpoint once and its position gets out-of-bound, break immediately.
            if x_reached_setpoint and not self.x_lower_bound < x < self.x_upper_bound:

                # print(f"{x:.3f} was out of bound (position)")

                # re-set PID to previous PID value
                self.PID = self.PID_last.copy()

                # restore original setup
                self.restore_setup()

                break
            
            # same for the object's angle
            if theta_reached_setpoint and not self.theta_lower_bound < theta < self.theta_upper_bound:

                # print(f"{theta:.3f} was out of bound (angle).")

                # re-set PID to previous PID value
                self.PID = self.PID_last.copy()

                # restore original setup
                self.restore_setup()

                break

            # calculate error, force
            error = self.desired_state - self.stepstate
            force = self.pidcontrol(error)

            # thus calculate new stepstate
            sol = integrate.odeint(self.pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity, self.fric_coef, self.fric_rot
            ))
            self.stepstate = np.array([sol[1][0], sol[1][1], sol[1][2], sol[1][3]])

            # check if cart-pole environment terminated (not the episode!)
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
        next_state = self.PID.copy()

        # controller was too unstable, and action was not [0] x 6
        # TODO: fix the second condition...action is not necessarily int anymore
        if list(self.PID_last) == list(self.PID) and list(action) != list(np.zeros(6, dtype=np.int64)):
            reward = -1
        
        # controller was stable, then calculate ISE improvements
        else:
            curr_ISE = ctut.calISE(trajectory, self.desired_state)

            # print if new best ISE is found 
            if curr_ISE < self.best_ISE: 
                self.best_ISE = curr_ISE.copy() 
                print(f"best ISE: {self.best_ISE:.2f} by {self.PID}")
            
            reward = (self.prev_ISE - curr_ISE) / 10
            self.prev_ISE = curr_ISE

        # episode ends after 50 PID updates
        truncated = True if self.time >= 50 else False
        self.time += 1       

        # terminated is always False
        return next_state, reward, False, truncated, [x_list, theta_list]  
    
    def step_offline(self, action: np.ndarray) -> tuple:

        '''
        For offline-tuning with linearized dynamics.
            - follows a very similar structure to gym's step() function

            - receives action and returns (s', r, terminated, truncated, info)
                - next_state (s') is simply current_state + action
                - reward is 5 * (curr_stability - prev_stability) (thus, the improvement in ISE scaled by 5).
                - terminated is always False
                - truncated is True after 50 time-steps
                - info is a list containing position, angle trajectory

            - action must be a np.ndarray of appropriate size (agreeing with control_mode)
        '''

        # action size must agree with control mode
        assert action.size == self.PID.size, f'action size ({action.size}) does not agree with PID size ({self.PID.size})'

        # reset env 
        self.iterreset()

        # calculate current stability
        self.prev_stability = self.get_curr_stability()

        # update and clip PID
        self.PID += action
        self.PID = self.clip_PID()

        # run simulation, record trajectory
        x_list, theta_list, trajectory = [], [], []
        for i in range(1000):

            # get new env stepstate
            x, x_dot, theta, theta_dot = self.stepstate
            x_list.append(x); theta_list.append(theta); trajectory.append(trajectory)

            # calculate error, force
            error = self.desired_state - self.stepstate
            force = self.pidcontrol(error)

            def pend(y, t, F, m_c, m_p, l_p, g):
                '''
                linearized-dynamics calculation for cartpole
                '''
                x, x_dot, th, th_dot = y
                
                xacc = ((3 * g * m_p)/(4 - 3 * m_p)) * th + (4 / ((m_p + m_c) * (4 - 3 * m_p))) * F

                thetaacc = ((3 * g * (m_p + m_c)) / (l_p * (4 - 3 * m_p))) * th + (3 / (l_p * (4 - 3 * m_p))) * F

                ytdt = [x_dot, xacc, th_dot, thetaacc]
                return ytdt

            sol = integrate.odeint(pend, [x, x_dot, theta, theta_dot], [0, self.tau], args = (
                float(force), self.masscart, self.masspole, self.length, self.gravity
            ))

            self.stepstate = (sol[1][0], sol[1][1], sol[1][2], sol[1][3])

            terminated = bool(
                sol[1][0] < -self.x_threshold
                or sol[1][0] > self.x_threshold
                or sol[1][2] < -self.theta_threshold_radians
                or sol[1][2] > self.theta_threshold_radians
            )

            if self.render_mode == "human":
                self.render()

            if terminated: break
        
        # calculate new stability and reward
        new_stability = self.get_curr_stability()
        reward = 5 * (new_stability - self.prev_stability) 
        self.prev_stability = new_stability

        # save good stability & PIDs
        if self.prev_stability > self.best_stability:
            self.best_stability = self.prev_stability
            self.best_PID = self.PID.copy()

            # print current best PID
            print(f"best stability: {self.best_stability:.2f} by {self.best_PID}")

        truncated = True if self.time >= 50 else False 
        self.time += 1

        return self.PID, reward, False, truncated, [x_list, theta_list]

    def pidcontrol(self, error: np.ndarray, use_diff_PID: np.ndarray = None) -> float:

        '''
        Calculates force from the error array using the PID controller. 
        error should be a np.ndarray of shape (4, ) calculated by (SP - stepstate).

            - set use_diff_PID to a custom PID vector if you want to temporarily use one.

        NOTE: The calculation process is dependent on control_mode (SISO or MIMO).

            if SISO:
            - PID should take the form of [K_p, K_i, K_d] <np.ndarray>
            - only the angle error term is taken into account

            if MIMO:
            - PID should take the form of [K_p_c, K_p_th, K_i_c, K_i_th, K_d_c, K_d_th] <np.ndarray>
            - both the angle and position error terms are considered.

        '''

        # use a custom PID if use_diff_PID is not None
        PID = self.PID if use_diff_PID is None else use_diff_PID


        # SISO
        if self.control_mode == 'pid1':
            error = error[2]
            mv = self.prev_mv + PID[0] * (error - self.prev_err) + self.tau * PID[1] * error + PID[2] * (error - 2*self.prev_err + self.dprev_err) / self.tau

        # MIMO
        elif self.control_mode == 'pid2':

            error = [error[0], error[2]]
        
            if self.prev_err == 0:
                self.prev_err = [0, 0]
                self.dprev_err = [0, 0]

                        
            mv = self.prev_mv + (np.dot(np.array([PID[0], PID[1]]), np.array(error) - np.array(self.prev_err)) 
                             + self.tau * np.dot(np.array([PID[2], PID[3]]), np.array(error)) 
                             + np.dot(np.array([[PID[4], PID[5]]]), np.array(error) - 2*np.array(self.prev_err) + np.array(self.dprev_err)) / self.tau)


        # clip force
        mv = np.clip(mv, -self.force_mag, self.force_mag)

        # update PID variables
        self.prev_mv = mv
        self.dprev_err = self.prev_err
        self.prev_err = error
        
        return mv

    def clip_PID(self) -> np.ndarray:

        '''
        - clips the magnitude of self.PID coefficients within (5, 150) and returns its copy
        - sign is preserved.
        '''

        # save the coefficients' signs
        signs = np.sign(self.PID)

        # take absolute value and then clip
        self.PID = np.clip(np.abs(self.PID), 5, 150)

        # give the signs back
        self.PID = signs * self.PID

        return self.PID.copy()

    def reset(self, *, custom_PID: np.ndarray = None, online: bool = False) -> np.ndarray:

        '''
        Resets the PID coefficients and returns a copy of it
            - resets the cart-pole environment to its randomized initial condition
            - for online tuning, returns the baseline PID with some noise
            - for offline tuning, returns a random PID vector from [5, 150]
            - must be called every new episode 
            - this allows the RL agent to explore the PID state space efficiently.
            - set custom_PID to a custom PID vector to start searching from there.
              
        '''

        # reset env
        super().reset(seed=None)

        # if no custom PID is given
        if custom_PID is None:

            # if online tuning, return baseline PID with noise
            if online:
                self.PID = self.PID_MIMO_BASELINE.copy() if self.control_mode=='pid2' else self.PID_SISO_BASELINE.copy()
                noise = np.random.uniform(low=-5, high=5, size=self.PID.size)
                self.PID = (self.PID + noise).copy()

            # if offline tuning, return random PID
            elif not online:
                self.PID = np.random.randint(low=5, high=150, size=3) if self.control_mode=='pid1' else \
                            np.random.randint(low=5, high=150, size=6) * np.array([1, -1, 1, -1, 1, -1])

        # if custom pid is given        
        elif custom_PID is not None:

            # check custom_PID's type, size
            assert isinstance(custom_PID, np.ndarray), f'expected PID vector to be of type np.ndarray but got {type(custom_PID)}'

            if self.control_mode == 'pid1':
                assert custom_PID.size == 3, f'expected PID vector of size 3 but got {custom_PID.size}'
            elif self.control_mode=='pid2': 
                assert custom_PID.size == 6, f'expected PID vector of size 6 but got {custom_PID.size}'
                
            self.PID = custom_PID.copy()

        # reset other stuff
        self.time = 0
        self.prev_stability = None

        if self.render_mode == "human":
            self.render()

        # return a copy of the initialized PID vector
        return self.PID.copy()

    def iterreset(self, custom: np.ndarray = None) -> None:

        '''
            - resets the cart-pole environment to its randomized initial condition
            - also resets the PID calculation variales 
            - must be called when switching to new PID values (that is, every time-step, and when calling 
              get_bound_by_rollout() and restore_setup())

        '''

        super().reset(seed=None)

        # get randomized initial env state
        low, high = utils.maybe_parse_reset_bounds(None, -0.05, 0.05)
        self.stepstate = self.np_random.uniform(low=low, high=high, size=(4,)) if custom is None else custom
        self.steps_beyond_terminated = None

        # reset PID variables
        self.integral = 0
        self.prev_err = 0
        self.prev_mv = 0

        return 

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