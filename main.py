from agent import *
from timeit import *
from user_env_gym import cartpolepid as cppid


'''
Create the cart-pole environment (custom)
    - set render_mode to 'human' for visualization
    - set control_mode to...
        - 'pid1' for SISO (single-input single-output, can only handle the pole's angle)
        - 'pid2' for MIMO (multi-input multi-output, can handle both the pole's angle & position)
'''
env = cppid.CartPoleEnv(render_mode='a', control_mode='pid2')


'''
Create the reinforcement learning (RL) agent
    - takes the environment as the input parameter, which must be created beforehand
'''
agent = Agent(env=env)


'''
Train the RL agent
    - the PID coefficients of the best-performing controllers will be printed out
    - set save to True to save the model under ./saved_models/
    - the training time will be kept track of and printed out via timeit (you can get rid of it, if you wish to)
'''
def t(): agent.train(save=True)
tt = timeit(stmt=lambda: t(), number=1)
print(f"---Training Completed in {tt:2f} seconds---")


