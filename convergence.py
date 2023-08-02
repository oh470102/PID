'''
finding a metric for convergence of x, theta
prints x, theta trajectories over time...

'''

from user_env_gym import cartpolepid as cppid
import numpy as np
import matplotlib.pyplot as plt

# create env
env = cppid.CartPoleEnv(render_mode='none', control_mode='pid2')

# set PID
PID = np.array([[-46, 130], [-25, 126], [-20, 16]])    # s = 1.14
# PID = np.array([[-46, 140], [-144, 16], [-111, 146]])   # s = 0.20
# PID = np.array([[-89, 133], [-137, 91], [-108, 83]])  # s = 0.40

# run simulation
env.iterreset(custom=np.array([0, 0, 0, 0]))
_, x, theta = env.step(PID)

env.iterreset(custom=np.array([0, 0, 0, 0]))
env.reset(custom_PID_MIMO=np.array([-89, 133, -137, 91, -108, 83]))
_, _, _, _, lis = env.linstep_MIMO(np.zeros(6, dtype=np.int64))
x_lin, theta_lin = tuple(lis)

# plot
fig, axs = plt.subplots(1, 2, figsize=(12, 9))

axs[0].plot(np.linspace(start=0, stop=len(theta), num=len(x)), x, label='pos_realdynamics')
axs[0].plot(np.linspace(start=0, stop=len(theta_lin), num=len(x_lin)), x_lin, label='pos_linear')
axs[0].set_xlabel('time')
axs[0].set_ylabel('position')

axs[1].plot(np.linspace(start=0, stop=len(theta), num=len(theta)), theta, label='angle_realdynamics')
axs[1].plot(np.linspace(start=0, stop=len(theta_lin), num=len(theta_lin)), theta_lin, label='angle_linear')
axs[1].set_xlabel('time')
axs[1].set_ylabel('angle')

# show plot
plt.tight_layout()
plt.legend(loc='best')
plt.savefig('traj_comparison_good.png')
plt.show()



