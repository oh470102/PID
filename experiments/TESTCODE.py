from user_env_gym import cartpolepid as cppid
import numpy as np

env = cppid.CartPoleEnv(render_mode='human', control_mode='pid2')

PID = np.array([[-46, 130], [-25, 126], [-20, 16]])

env.iterreset(custom=np.array(list((-0.09687904360928577, -0.018039169416338, -0.20355052400554438, -0.7424110390573255))))
env.restore_setup()