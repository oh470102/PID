import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import matlab.engine as mat
import matlab
from user_env_gym import controlutil as ctut

from scipy import integrate

class SOPDTenv():

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # system information
        self.input_time_delay = 2
        self.den_coef = [25, 10, 1]
        self.num_coef = 0.3

        # control information
        self.force_mag = 50

        # simulation information
        self.tau = 0.02
        self.stepstate = None
        self.input_queue = [0] * (int)(self.input_time_delay / self.tau)

        # temporary variables in order to implement digital pid (using velocity form)
        self.prev_mv = 0
        self.prev_err = 0
        self.dprev_err = 0
        self.setpoint = 5
        self.time = 0 
        self.control_mode = 'pid1'

        # MATLAB engine (for stability calculation)
        self.eng = ctut.start_matengine()

        # PID stuff (for reward calculation)
        self.PID = None
        self.PID_last = None
        self.PID_SISO_BASELINE = np.array([16.62605434 , 3.3202569 , 35.7224133 ])
        # More PID stuff (for saving best-performers)
        self.best_stability = None
        self.prev_stability = None
        self.best_PID = None
        self.best_ISE = None    
        self.PID_BOUND = 100

        # for stability-wise guarantees
        self.lin_stability_threshold = 0.05
        self.y_upper_bound, self.y_lower_bound, self.prev_ISE = self.get_bound_by_rollout()
        print(f"Bound: {self.y_lower_bound, self.y_upper_bound} :: Baseline ISE: {self.prev_ISE}")
        self.best_ISE = self.prev_ISE

    def get_curr_stability(self) -> float:

        '''
        returns the current PID coefficients' stability
            - the sign is reversed, so the more positive, the more stable
            - NOTE: stability calculation depends on the control_mode
        '''

        MIMO = False

        if not MIMO:
            P, I, D = tuple(map(float, list(self.PID)))

            A, B, C, DD, E = ctut.sopdt_dss()
            stability = ctut.lin_stab_td_SISO(self.eng, P, I, D, A, B, C, DD, E, intd = 2)

            return -stability

    def restore_setup(self) -> None:

        '''
        Restores the original cart-pole setup 
        when the controller becomes too unstable.
        Baseline PID is used for this process.
        '''

        # SP is now the initial condition (around 0)
        restoration_setpoint = 0

        # reset PID variables 
        # NOTE: do NOT call iterreset() since that automatically brings the cart to the initial condition
        self.prev_mv = 0
        self.prev_err = 0
        self.dprev_err = 0
        self.input_queue = [0] * (int)(self.input_time_delay / self.tau)

        # use baseline PID
        PID = self.PID_SISO_BASELINE

        # print("Restoration initiated...   ------>    ", end='')

        # loop until initial condition restored
        i = 0
        while True:

            # retrieve current env state variables
            y, y_dot = tuple(self.stepstate)

            # calculate error, force
            error = restoration_setpoint - y
            input = self.pidcontrol(error, use_diff_PID=PID)
            self.input_queue.append(input)

            # break if error is sufficiently small for both position & angle (initial state is restored
            if abs(error) < 0.025: 
                # print('Succeeded!')
                break

            # otherwise, calculate new env stepstates
            curinput = self.input_queue.pop(0)
            sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
                curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
            ))
            self.stepstate = (sol[1][0], sol[1][1])

            # termiation condition
            i += 1
            if i > 2_000: print("Fatal: restoration process failed by taking too long.")

    def get_bound_by_rollout(self, generosity: float = 0.35) -> tuple:

        '''
        calculates upper and lower bounds of state variable(s) -> y
        through a rollout (simulation) of baseline PID

            - larger generosity means larger bound
            - also returns ISE of baseline PID for future reward calculation
        '''

        # reset env (may change to random)
        self.reset()

        # setup variables
        rollout_traj_y, input_list, output_list = [], [] ,[]
        reached_setpoint_at = [None]
 
        # use baseline PID
        PID = self.PID_SISO_BASELINE

        # env loop
        # NOTE: we know baseline PID will not fail, so environment termination condition was not considered.
        for i in range(10000):
            
            # record current state variables (x, v, theta, w)
            y, y_dot = self.stepstate
            rollout_traj_y.append(y)
            
            # calculate error, force
            error = self.setpoint - y
            input = self.pidcontrol(self.setpoint - y, use_diff_PID=PID)
            self.input_queue.append(input)

            # record the instance at which setpoint was reached
            if reached_setpoint_at[0] is not None and abs(error) < self.setpoint * 1e-2: reached_setpoint_at[0] = i

            # calculate new state variables 
            curinput = self.input_queue.pop(0)
            sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
                curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
            ))
            self.stepstate = (sol[1][0], sol[1][1])

            # record input/output
            input_list.append(input)
            output_list.append(self.stepstate[0])
        
        # calculate upper, lower bounds for y
        rollout_traj_y = np.array(rollout_traj_y[reached_setpoint_at[0]: ])
        max_y = np.max(rollout_traj_y)
        y_upper_bound = max_y + generosity * (max_y - self.setpoint)
        y_lower_bound = self.setpoint - (y_upper_bound - self.setpoint)
        baseline_ISE = ctut.calISE(output_list, self.setpoint, ITSE=True)

        plt.plot(output_list)
        plt.grid()
        plt.show()

        return y_upper_bound, y_lower_bound, baseline_ISE

    def pend(self, state, t, x, a_2, a_1, a_0, b):
        y, y_dot = state

        y_ddot = (1 / a_2) * (b * x - a_1 * y_dot - a_0 * y)

        stdt = [y_dot, y_ddot]
        return stdt

    def step(self, action):
        self.reset()

        input_list = []
        output_list = []

        for i in range(3000):
            y, y_dot = self.stepstate

            input = self.pidcontrol(self.setpoint - y)
            self.input_queue.append(input)
            
            curinput = self.input_queue.pop(0)
            
            sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
                curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
            ))

            self.stepstate = (sol[1][0], sol[1][1])

            input_list.append(input)
            output_list.append(self.stepstate[0])
        
        return [], [], [], [], [input_list, output_list]

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
        self.reset()

        # calculate current stability
        self.prev_stability = self.get_curr_stability()

        # update and clip PID
        self.PID += action
        self.PID = self.clip_PID()

        # run simulation, record trajectory
        input_list, output_list = [], []

        # for i in range(1000):

        #     # get new stepstate
        #     y, y_dot = self.stepstate

        #     # calculate error, force
        #     error = self.setpoint - y
        #     input = self.pidcontrol(error)
        #     self.input_queue.append(input)

        #     # calculate new state variables 
        #     curinput = self.input_queue.pop(0)
        #     sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
        #         curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
        #     ))
        #     self.stepstate = (sol[1][0], sol[1][1])

        #     # record input/output
        #     input_list.append(input)
        #     output_list.append(self.stepstate[0])

        #     terminated = bool(i == 4999)
        #     if terminated: break
        
        # calculate new stability and reward
        new_stability = self.get_curr_stability()
        reward = 50 * (new_stability - self.prev_stability) 
        self.prev_stability = new_stability

        # save good stability & PIDs
        if self.best_stability is None or self.prev_stability > self.best_stability:
            self.best_stability = self.prev_stability
            self.best_PID = self.PID.copy()

            # print current best PID
            print(f"best stability: {self.best_stability:.2f} by {self.best_PID}")

        truncated = True if self.time >= 50 else False 
        self.time += 1

        return self.PID, reward, False, truncated, [input_list, output_list]
    
    def step_online(self, action: np.ndarray) -> tuple:

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
        self.reset()

        # save current PID 
        self.PID_last = self.PID.copy()

        # update & clip PID by action
        self.PID += action
        self.PID = self.clip_PID()

        # record trajectory to keep track of controller stability
        y_reached_setpoint = False
        input_list, output_list = [], []
        reward = 0

        for i in range(10000):
            
            # calculate linear stability, just once prior to simulation
            if i == 0:
                lin_stability = self.get_curr_stability()

                # if too unstable, break immediately.
                if lin_stability < self.lin_stability_threshold:

                    # print("Too unstable!")
                
                    # re-set PID to previous PID value
                    self.PID = self.PID_last.copy()

                    break

            # record trajectory
            y, y_dot = self.stepstate

            # check whether x, theta is near setpoint
            if y_reached_setpoint is False and abs(y - self.setpoint) < self.setpoint * 1e-2: y_reached_setpoint = True

            # if object reached setpoint once and its position gets out-of-bound, break immediately.
            if y_reached_setpoint and not self.y_lower_bound < y < self.y_upper_bound:

                # print(f"{y:.3f} was out of bound (y)")

                # re-set PID to previous PID value
                self.PID = self.PID_last.copy()

                # restore original setup
                self.restore_setup()

                break
        
            # calculate error, force
            error = self.setpoint - y
            input = self.pidcontrol(error)
            self.input_queue.append(input)

            # calculate new state variables 
            curinput = self.input_queue.pop(0)
            sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
                curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
            ))
            self.stepstate = (sol[1][0], sol[1][1])

            # record input/output
            input_list.append(input)
            output_list.append(self.stepstate[0])

        # set next_state as updated PID 
        next_state = self.PID.copy()

        # controller was too unstable, and action was not [0] x 6
        if list(self.PID_last) == list(self.PID): 
            reward = -1
        
        # controller was stable, then calculate ISE improvements
        else:
            curr_ISE = ctut.calISE(output_list, self.setpoint, ITSE=True)
            # print(f"{curr_ISE} by {self.PID}")

            # print if new best ISE is found 
            if curr_ISE < self.best_ISE: 
                self.best_ISE = curr_ISE.copy() 
                print(f"best ISE: {self.best_ISE:.2f} by {self.PID}")
            
            reward = (self.prev_ISE - curr_ISE)/1e6
            self.prev_ISE = curr_ISE

        # episode ends after 50 PID updates
        truncated = True if self.time >= 50 else False
        self.time += 1       

        # terminated is always False
        return next_state, reward, False, truncated, [input_list, output_list]  

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
            mv = self.prev_mv + PID[0] * (error - self.prev_err) + self.tau * PID[1] * error + PID[2] * (error - 2*self.prev_err + self.dprev_err) / self.tau

        # clip force
        mv = np.clip(mv, -self.force_mag, self.force_mag)

        # update PID variables
        self.prev_mv = mv
        self.dprev_err = self.prev_err
        self.prev_err = error
        
        return mv
    
    def reset(self):
        self.stepstate = [0, 0] # y, y_dot
        self.prev_err = 0
        self.prev_mv = 0
        self.input_queue = [0] * (int)(self.input_time_delay / self.tau)
    
    def reset_PID(self, custom_PID: np.ndarray = None, online: bool = False) -> np.ndarray:

        '''
        Resets the PID coefficients and returns a copy of it
            - resets the cart-pole environment to its randomized initial condition
            - for online tuning, returns the baseline PID with some noise
            - for offline tuning, returns a random PID vector from [5, 150]
            - must be called every new episode 
            - this allows the RL agent to explore the PID state space efficiently.
            - set custom_PID to a custom PID vector to start searching from there.
              
        '''

        # if no custom PID is given
        if custom_PID is None:

            # if online tuning, return baseline PID with noise
            if online:
                self.PID = self.PID_SISO_BASELINE.copy()
                noise = np.random.uniform(low=-self.PID_BOUND/20, high=self.PID_BOUND/20, size=self.PID.size)
                self.PID = (self.PID + noise).copy()

            # if offline tuning, return random PID
            elif not online:
                self.PID = np.random.uniform(low=0, high=self.PID_BOUND, size=3) if self.control_mode=='pid1' else \
                            np.random.uniform(low=0, high=self.PID_BOUND, size=6) * np.array([1, -1, 1, -1, 1, -1])

        # if custom pid is given        
        elif custom_PID is not None:

            # check custom_PID's type, size
            assert isinstance(custom_PID, np.ndarray), f'expected PID vector to be of type np.ndarray but got {type(custom_PID)}'
            assert custom_PID.size == 3, f'expected PID vector of size 3 but got {custom_PID.size}'

            self.PID = custom_PID.copy()

        # reset other stuff
        self.time = 0
        self.prev_stability = None

        # return a copy of the initialized PID vector
        return np.array(self.PID.copy(), dtype=np.float64)
    
    def clip_PID(self) -> np.ndarray:

        '''
        - clips the magnitude of self.PID coefficients within (5, 150) and returns its copy
        - sign is preserved.
        '''

        if self.control_mode=='pid2':

            # save the coefficients' signs
            signs = np.sign(self.PID)

            # take absolute value and then clip
            self.PID = np.clip(np.abs(self.PID), 0, self.PID_BOUND)

            # give the signs back
            self.PID = signs * self.PID

        elif self.control_mode=='pid1':

            self.PID = np.clip(self.PID, 0, self.PID_BOUND)

        return self.PID.copy()