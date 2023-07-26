import user_env_gym.cartpolepid as cppid
import user_env_gym.controlutil as ctut

As = [[0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.7945946, 0.0, 0.0],
    [0.0, 17.4810811, 0.0, 0.0]]
Bs = [[0],
    [0],
    [0.982801],
    [1.62162]]
Cs = [0, 1, 0, 0]
Ds = [0]
Es = [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]

Am = [[0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0.7945946, 0, 0],
    [0, 17.4810811, 0, 0]]
Bm = [[0],
    [0],
    [0.982801],
    [1.62162]]
Cm = [[1, 0, 0, 0],
    [0, 1, 0, 0]]
Dm = [[0],
    [0]]
Em = [[1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]


# ctut.lin_stability_SISO(100, 100, 10, As, Bs, Cs, Ds, Es)
# ctut.lin_stability_MIMO([-20,100],[-10, 100],[0,10], Am, Bm, Cm, Dm, Em)

eng = ctut.start_matengine()
import numpy as np

# test SISO
# while True:
#     env = cppid.CartPoleEnv(render_mode='human', control_mode= 'pid1')
#     custom_PID = np.array(list(map(int, input("ENTER CUSTOM PID: ").split())))
#     PID, _ = env.reset(custom_PID=custom_PID)
#     P, I, D = tuple(map(int, tuple(PID)))

#     for _ in range(3):
#         env.linstep(action=np.array([0, 0, 0]))
        
#     print(f"stability: {ctut.lin_stability_SISO(eng, P, I, D, As, Bs, Cs, Ds, Es)}")
#     env.close()

# test MIMO
while True:
    env = cppid.CartPoleEnv(render_mode='human', control_mode= 'pid2')
    custom_PID_MIMO = np.array(list(map(int, input("ENTER CUSTOM PID: ").split())))
    PID_MIMO, _ = env.reset(custom_PID_MIMO=custom_PID_MIMO, MIMO=True)
    P1, P2, I1, I2, D1, D2 = tuple(map(int, tuple(PID_MIMO)))

    for _ in range(3):
        env.linstep_MIMO(action=np.array([0, 0, 0, 0, 0, 0]))
        
    print(f"stability: {ctut.lin_stability_MIMO(eng, [P1, P2],[I1, I2],[D1,D2], Am, Bm, Cm, Dm, Em)}")
    env.close()