from user_env_gym import cartpolepid as cppid, sopdtpid as spid, tankpid as tpid
import numpy as np, matplotlib.pyplot as plt
import seaborn as sns

'''

Test any PID coefficient!
'''

''' 
Cart-pole 

env = cppid.CartPoleEnv(render_mode='human', control_mode='pid2')
env.reset(custom_PID=PID)
env.iterreset(custom=np.array([0,0,0,0]))
env.step_online(action=np.array([0,0,0,0,0,0]))
'''

'''
Tank Env

env = tpid.TankEnv()

# PID 1
env.reset_PID(custom_PID=np.array([6.608, 80, 0]))
_, _, _, _, lis1 = env.step_online(action=np.array([0,0,0]))
env.reset()

# PID 2
env.reset_PID(custom_PID=np.array([38.52332521, 100,           1.35016392]))
_, _, _, _, lis2 = env.step_online(action=np.array([0,0,0]))
env.reset()

# PID 3
env.reset_PID(custom_PID=np.array([ 37.00028898, 100.   ,        0.        ]))
_, _, _, _, lis3 = env.step_online(action=np.array([0,0,0]))
env.reset()
'''


# SOPDT
env = spid.SOPDTenv()

# PID 1
env.reset_PID(custom_PID=np.array([13.95, 1.885, 25.81]))
_, _, _, _, lis1 = env.step(action=np.array([0,0,0]))
env.reset()

# PID 2
env.reset_PID(custom_PID=np.array([16.51, 1.983, 21.43]))
_, _, _, _, lis2 = env.step(action=np.array([0,0,0]))
env.reset()

# PID 3
env.reset_PID(custom_PID=np.array([11.05, 0.612, 19.22]))
_, _, _, _, lis3 = env.step(action=np.array([0,0,0]))
env.reset()

# PID 4, ours
env.reset_PID(custom_PID=np.array([20.25064878,  4.33564473, 40.61529733]))
_, _, _, _, lis4 = env.step(action=np.array([0,0,0]))
env.reset()

# PID 5
env.reset_PID(custom_PID=np.array([6.98, 0.541, 12.91]))
_, _, _, _, lis5 = env.step(action=np.array([0,0,0]))
env.reset()


plt.plot(lis1[1], label='Z-N')
plt.plot(lis2[1], label='C-C')
plt.plot(lis3[1], label='CHR 20%')
plt.plot(lis4[1], label='Ours')
plt.plot(lis5[1], label='CHR 0%')

plt.legend()
plt.grid()
plt.savefig('./imgs/sopdt/traj.png')
plt.show()



