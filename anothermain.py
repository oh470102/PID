import user_env_gym.cartpolepid as cppid
from utils import *
from DQNagent import *

### resolve matplotlib error
resolve_matplotlib_error()
plt.ion()

### create env
env = cppid.CartPoleEnv(render_mode=None, control_mode= 'pid1')

# initialize agent, set epoch length
agent = DQNAgent(env=env)

# train the agent if not load
epochs = 2000#int(input("EPOCHS: "))
save = False#True if input("Save Agent? (True or False): ") == 'True' else False
scores = agent.train(epochs)
print("------Training Completed------")

# turn off plotting interactive mode
plt.ioff()
plt.plot(agent.save_epi_reward)

# plot moving average 
final_plot(scores)
analyze(scores)

#see how the trained agent performs
showcase(agent=agent, env=cppid.CartPoleEnv(render_mode='human', control_mode= 'pid1'))
