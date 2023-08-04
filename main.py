from agent import *
from timeit import *
from user_env_gym import cartpolepid as cppid

# create env
env = cppid.CartPoleEnv(render_mode=None, control_mode='pid2')

# create agent
agent = Agent(env=env)

# train agent
def t(): agent.train(save=True)
tt = timeit(stmt=lambda: t(), number=1)
print(f"---Training Completed in {tt:2f} seconds---")

# see performance
agent.test_agent(cppid.CartPoleEnv(render_mode='none', control_mode='pid2'), MIMO=True)
