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
PID = np.array([[-55, 134], [-40, 127], [-16, 14]])             # ISE = 609
PID_baseline = np.array([[-57, 103], [-34, 87], [-13,  16]])   # baseline, ISE = 566
# PID = np.array([[-46, 140], [-144, 16], [-111, 146]])   # s = 0.20
# PID = np.array([[-89, 133], [-137, 91], [-108, 83]])  # s = 0.40

# run simulation
# env.iterreset(custom=np.array([0, 0, 0, 0]))
env.iterreset()
_, x, v, theta, w = env.step(PID)

# env.iterreset(custom=np.array([0, 0, 0, 0]))
env.iterreset()
_, x_baseline, v_baseline, theta_baseline, w_baseline = env.step(PID_baseline)


# env.iterreset(custom=np.array([0, 0, 0, 0]))
env.iterreset()
env.reset(custom_PID_MIMO=np.array([-46, 130, -25, 126, -20, 16]))
_, _, _, _, lis = env.linstep_MIMO(np.zeros(6, dtype=np.int64))
x_lin, v_lin, theta_lin, w_lin = tuple(lis)

# plot
fig, axs = plt.subplots(1, 4, figsize=(12, 9))

axs[0].plot(np.linspace(start=0, stop=len(x), num=len(x)), x, label='ISE=609')
axs[0].plot(np.linspace(start=0, stop=len(x_lin), num=len(x_lin)), x_lin, label='pos_linear')
axs[0].plot(np.linspace(start=0, stop=len(x_baseline), num=len(x_baseline)), x_baseline, label='ISE=566')
axs[0].set_xlabel('time')
axs[0].set_ylabel('position')
axs[0].legend()

axs[1].plot(np.linspace(start=0, stop=len(theta), num=len(theta)), theta, label='ISE=609')
axs[1].plot(np.linspace(start=0, stop=len(theta_lin), num=len(theta_lin)), theta_lin, label='pos_linear')
axs[1].plot(np.linspace(start=0, stop=len(theta_baseline), num=len(theta_baseline)), theta_baseline, label='ISE=566')
axs[1].set_xlabel('time')
axs[1].set_ylabel('angle')
axs[1].legend()

axs[2].plot(np.linspace(start=0, stop=len(v), num=len(v)), v, label='ISE=609')
axs[2].plot(np.linspace(start=0, stop=len(v_lin), num=len(v_lin)), v_lin, label='pos_linear')
axs[2].plot(np.linspace(start=0, stop=len(v_baseline), num=len(v_baseline)), v_baseline, label='ISE=566')
axs[2].set_xlabel('time')
axs[2].set_ylabel('velocity')
axs[2].legend()

axs[3].plot(np.linspace(start=0, stop=len(w), num=len(w)), w, label='ISE=609')
axs[3].plot(np.linspace(start=0, stop=len(w_lin), num=len(w_lin)), w_lin, label='angle_linear')
axs[3].plot(np.linspace(start=0, stop=len(w_baseline), num=len(w_baseline)), w_baseline, label='ISE=566')
axs[3].set_xlabel('time')
axs[3].set_ylabel('angular velocity')
axs[3].legend()

# show plot
plt.tight_layout()
plt.savefig('ISE=566 and 609 with random resets.png')
plt.show()



