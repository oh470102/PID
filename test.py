import user_env_gym.cartpolepid as cppid
import matplotlib.pyplot as plt
import numpy as np
import math

env = cppid.CartPoleEnv(control_mode= 'pid1', render_mode='human')

for i in range(10):
    #env.reset()
    # state, reward, info = env.step([(math.sqrt(23.52) + 2) * 5, (math.sqrt(23.52) * 2) * 5, (1) * 5])

    state, reward, info = env.linstep(np.array([100,100,10]))

    # state, reward, info = env.linstep([[20,100],[10,100],[0,10]])

    # state, reward, _ = env.step([[(math.sqrt(23.52) + 2) * 0.9, (math.sqrt(23.52) + 2) * 5], [(math.sqrt(23.52) * 2) * 0.9, (math.sqrt(23.52) * 2) * 5], [(1) * 0.9, (1) * 5]])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    # it = 2
    # kpmax = 50
    # kimax = 70
    # kdmax = 10
    # while True:
        
    #     it += 1
    #     tmp = 2**it
    #     x = []
    #     y = []
    #     z = []
    #     c = []
    #     for i in range(tmp):
    #         for j in range(tmp):
    #             for k in range(tmp):
    #                 kp = 50 * (2 * i + 1) / (2 * tmp)
    #                 ki = 70 * (2 * j + 1) / (2 * tmp)
    #                 kd = 10 * (2 * k + 1) / (2 * tmp)
    #                 score, info = env.coefstep([kp,ki,kd])
    #                 x.append(kp)
    #                 y.append(ki)
    #                 z.append(kd)
    #                 c.append([(1 - score/8), (score / 8), 0])
    #                 print(i)
    #     ax.scatter(x,y,z, c = c)

    #     plt.show()



        
    # break;
    
    #print(reward)