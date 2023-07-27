from user_env_gym import cartpolepid as cppid
import numpy as np

# create env
env = cppid.CartPoleEnv(render_mode='human', control_mode='pid1')

# generate PID list
PID1 = np.array([100, 80, 10]) # s = 1.00
PID2 = np.array([80, 103, 10]) # s = 2.00
PID3 = np.array([72, 110, 10]) # s = 3.00
PID4 = np.array([70, 116, 10]) # s = 4.00
PID5 = np.array([72, 136, 10]) # s = 5.36
PID_list = [PID1, PID2, PID3, PID4, PID5]
PID_dict = {f"PID{i}":[] for i in range(1, len(PID_list)+1)}


# test PIDs
for i, PID in enumerate(PID_list, start=1):

    for _ in range(5):
        custom_init_state = np.array(list(map(float, input("Enter custom init state: ").split())))
        env.iterreset(custom=custom_init_state)
        reward, init_state = env.step(PID)
        PID_dict[f"PID{i}"].append(reward) 

# close env
env.close()

# print results
print(PID_dict)
