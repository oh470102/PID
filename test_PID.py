from user_env_gym import cartpolepid as cppid
import numpy as np

'''
Test any PID coefficient!
'''

PID = np.array([-92.46397335, 150,         -79.52151719 , 79.39413694, -42.29593014,
  43.68005843])
# PID = np.array([-57, 103, -34,  87, -13,  16])

env = cppid.CartPoleEnv(render_mode='human', control_mode='pid2')
env.reset(custom_PID=PID)
env.iterreset(custom=np.array([0,0,0,0]))

env.step_online(action=np.array([0,0,0,0,0,0]))




