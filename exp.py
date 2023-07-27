from user_env_gym import cartpolepid as cppid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# create env
env = cppid.CartPoleEnv(render_mode='none', control_mode='pid1')

# generate PID list
PID1 = np.array([24, 100, 5])
PID2 = np.array([24, 77, 5])
PID3 = np.array([26, 70, 5])
PID4 = np.array([110, 77, 5])
# PID1 = np.array([100, 80, 10]) # s = 1.00
# PID2 = np.array([80, 103, 10]) # s = 2.00
# PID3 = np.array([72, 110, 10]) # s = 3.00
# PID4 = np.array([70, 116, 10]) # s = 4.00
PID5 = np.array([72, 136, 10]) # s = 5.36
PID_list = [PID1, PID2, PID3, PID4, PID5]
PID_dict = {f"PID{i}":[] for i in range(1, len(PID_list)+1)}

# test PIDs
for i, PID in tqdm(enumerate(PID_list, start=1)):

    angle_bound = 100

    for j in range(-angle_bound, angle_bound+1):

        rewards = []

        for _ in range(5):
            # set custom initial state
            custom_init_state = np.array([0, 0, 0.01*j, 0])

            # reset state
            env.iterreset(custom=custom_init_state)
            
            # step, record results
            reward, init_state = env.step(PID)
            rewards.append(reward)

        PID_dict[f"PID{i}"].append(sum(rewards)/len(rewards)) 

# close env
env.close()

# display results
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
i = 0

for scores in PID_dict.values():

    # find boundary
    j = 0
    
    while scores[j] < 500 and j < 200: 
        try:
            j += 1
        except: print(j)

    # put them on a graph
    x = np.linspace(-1, 1, 201)
    axs[i//3, i%3].plot(x, scores, label=-round(math.degrees(x[j]), 2))
    
    # label
    axs[i//3, i%3].set_title(f'PID{i+1}')
    axs[i//3, i%3].set_xlabel('radians')
    axs[i//3, i%3].set_ylabel('score')
    axs[i//3, i%3].legend(loc='best')

    # indexing
    i += 1

plt.tight_layout()
plt.savefig('noise=9.png')
plt.show()
plt.close()
