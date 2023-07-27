from agent import *
from timeit import *
from user_env_gym import cartpolepid as cppid

# create env
env = cppid.CartPoleEnv(render_mode=None, control_mode='pid1')

# create agent
agent = Agent(env=env)

# train agent
save = True
t = timeit(stmt=agent.train(save=save), number=1)
print(f"---Training Completed in {t:2f} seconds---")

# see performance
agent.test_agent(cppid.CartPoleEnv(render_mode='human', control_mode='pid1'))
