from user_env_gym import cartpolepid as cppid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

# create env
env = cppid.CartPoleEnv(render_mode='none', control_mode='pid2')

# generate PID list
PID1 = np.array([[-46, 140], [-144, 16], [-111, 146]]) # s = 0.20
PID2 = np.array([[-89, 133], [-137, 91], [-108, 83]])  # s = 0.40
PID3 = np.array([[-90, 126], [-118, 146], [-47, 40]])  # s = 0.65
PID4 = np.array([[-72, 149], [-58, 108], [-45, 45]])   # s = 0.80
PID5 = np.array([[-46, 130], [-25, 126], [-20, 16]])   # s = 1.14

PID_list = [PID1, PID2, PID3, PID4, PID5]
PID_dict = {f"PID{i}":[] for i in range(1, len(PID_list)+1)}

# test PIDs
for i, PID in tqdm(enumerate(PID_list, start=1)):

    angle_bound = 100

    for j in range(-angle_bound, angle_bound+1):

        rewards = []

        for _ in range(7):
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
fig.suptitle('noise=0')
i = 0

for scores in PID_dict.values():

    # find boundary
    j = 0
    
    while scores[j] < 500 and j < 200: 
        j += 1

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