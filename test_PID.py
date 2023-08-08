from user_env_gym import cartpolepid as cppid
import numpy as np

'''
Test any PID coefficient!
'''

PID = np.array([-70.86045582, 123.30184012, -60.15679048, 109.16208843, -29.05265625, 33.42436623])
# PID = np.array([-57, 103, -34,  87, -13,  16])

env = cppid.CartPoleEnv(render_mode='human', control_mode='pid2')
env.reset(custom_PID=PID)
env.iterreset(custom=np.array([0,0,0,0]))

env.step_online(action=np.array([0,0,0,0,0,0]))




