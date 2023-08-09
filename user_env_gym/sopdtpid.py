import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import matlab.engine as mat
import matlab

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

    def pend(self, state, t, x, a_2, a_1, a_0, b):
        y, y_dot = state

        y_ddot = (1 / a_2) * (b * x - a_1 * y_dot - a_0 * y)

        stdt = [y_dot, y_ddot]
        return stdt

    def step(self, pid):
        self.reset()
        setpoint = 10

        input_list = []
        output_list = []

        for i in range(10000):
            y, y_dot = self.stepstate

            input = self.pidcontrol(setpoint - y, pid)
            self.input_queue.append(input)
            
            curinput = self.input_queue.pop(0)
            
            sol = integrate.odeint(self.pend, [y, y_dot], [0, self.tau], args = (
                curinput, self.den_coef[0], self.den_coef[1], self.den_coef[2], self.num_coef
            ))

            self.stepstate = (sol[1][0], sol[1][1])

            input_list.append(input)
            output_list.append(self.stepstate[0])
        
        return input_list, output_list


    def pidcontrol(self, error, action):
        # action should be [K_p, K_i, K_d]
        mv = self.prev_mv + action[0] * (error - self.prev_err) + self.tau * action[1] * error + action[2] * (error - 2*self.prev_err + self.dprev_err) / self.tau

        if mv > self.force_mag:
            mv = self.force_mag
        
        if mv < -self.force_mag:
            mv = -self.force_mag

        self.prev_mv = mv
        self.dprev_err = self.prev_err
        self.prev_err = error
        
        return mv
    
    def reset(self):
        self.stepstate = [0, 0] # y, y_dot
        self.prev_err = 0
        self.prev_mv = 0
