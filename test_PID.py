from user_env_gym import cartpolepid as cppid, sopdtpid as spid, tankpid as tpid
import numpy as np, matplotlib.pyplot as plt
import seaborn as sns

'''

Test any PID coefficient!
'''


# Cart-pole 

env = cppid.CartPoleEnv(render_mode='humaan', control_mode='pid2')

# PID 1
PID = np.array([-20, 106, -37, 136, -66, 129.]) # baseline
env.reset(custom_PID=PID, online=True)
x, theta, rx, rtheta = env.step_online(action=np.array([0,0,0,0,0,0]))


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
axes[0].plot([i for i in range(0, len(x))], x, label='random PID')
axes[0].plot([i for i in range(len(x)-1, len(x) + len(rx)-1)], rx, label='Baseline PID')
axes[0].set_title('Position Restoration Trajectory')
axes[0].legend()
axes[0].grid(True)

axes[1].plot([i for i in range (0, len(theta))], theta, label='random PID')
axes[1].plot([i for i in range(len(theta)-1, len(theta) + len(rtheta)-1)], rtheta, label='Baseline PID')
axes[1].set_title('Angle Restoration Trajectory')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Cartpole Env. Restoration Trajectory')
plt.savefig('./imgs/cartpole/restoration_trajectory.png')
plt.show()



