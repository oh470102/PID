'''
finding a metric for convergence of x, theta
'''

from user_env_gym import cartpolepid as cppid
import numpy as np
import matplotlib.pyplot as plt

# create env
env = cppid.CartPoleEnv(render_mode='none', control_mode='pid2')

# set PID
PID_good = np.array([[-46, 130], [-25, 126], [-20, 16]])    # s = 1.14
PID_bad = np.array([[-46, 140], [-144, 16], [-111, 146]])   # s = 0.20

# run simulation
_, x, theta = env.step(PID_bad)

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 9))

axs[0].plot(np.linspace(start=0, stop=len(theta), num=len(x)), x, label='position')
axs[0].set_xlabel('time')
axs[0].set_ylabel('position')

axs[1].plot(np.linspace(start=0, stop=len(theta), num=len(theta)), theta, label='angle')
axs[1].set_xlabel('time')
axs[1].set_ylabel('angle')

# show plot
plt.tight_layout()
plt.show()

