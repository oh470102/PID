from networks import *
from tqdm import tqdm
from collections import deque
import numpy as np, copy, random

class Agent:
    def __init__(self, env):
        self.GAMMA = 0.99
        self.NUM_EPISODES = 1_000
        self.SOLVED_SCORE = 200

        self.BATCH_SIZE = 200
        self.SYNCH_FREQ = 500
        self.MEM_SIZE = 100_000

        self.env = env
        self.state_dim = 6
        self.action_dim = 729 # 3**6

        self.actor = model(self.state_dim, self.action_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.replay = deque(maxlen=self.MEM_SIZE)
        self.scores = []

        self.epsilon = 1.0
        self.epsilon_checked = False

        self.P, self.I, self.D = [], [], []

        self.action_map = [[a,b,c,d,e,f] for a in [1, -1, 0] for b in [1, -1, 0] for c in [1, -1, 0] for d in [1, -1, 0] for e in [1, -1, 0] for f in [1, -1, 0]]

    def get_action(self, state):

        q = self.actor(torch.from_numpy(state).float())
        q_ = q.detach().numpy()

        if random.random() < self.epsilon:
            action = np.random.randint(0, int(self.action_dim))
        else:
            action = np.argmax(q_)

        return action

    def train(self, mp=False, save=False):

        for ep in tqdm(range(self.NUM_EPISODES)):

            # reset and retrieve initial PID
            state, _ = self.env.reset(MIMO=True, online=True) 
            done, score, synch_i = False, 0, 0

            # save initial PID for comparison
            saved_init_state = state

            while not done:
                
                synch_i += 1
                if synch_i % self.SYNCH_FREQ == 0:
                    self.target_actor.load_state_dict(self.actor.state_dict())

                action = self.get_action(state)

                # print(f"------on episode {ep}, time {synch_i-1}------")
                # give some noise to action
                next_state, reward, term, trunc, _ = self.env.step_online(np.array(self.action_map[action]) + np.random.normal(loc=0, scale=0.5, size=6))
                done = term or trunc
                exp = (torch.tensor(state), action, reward, torch.tensor(next_state), done)
                self.replay.append(exp)
                score += reward

                # if synch_i % 10 == 0:
                #     print(f"initial PID: {state}")
                #     print(f"Action: {self.action_map[action]}")
                #     print(f"reward: {reward}")
                #     print(f"next PID {next_state}")

                state = next_state

                if len(self.replay) > self.BATCH_SIZE:
                    minibatch = random.sample(self.replay, self.BATCH_SIZE)
                    state_batch = torch.stack([s1 for s1, a, r, s2, d in minibatch])
                    action_batch = torch.Tensor([a for s1, a, r, s2, d in minibatch])
                    reward_batch = torch.Tensor([r for s1, a, r, s2, d in minibatch]).reshape(-1, 1)
                    state2_batch = torch.stack([s2 for s1, a, r, s2, d in minibatch])
                    done_batch = torch.Tensor([d for s1, a, r, s2, d in minibatch]).reshape(-1, 1)

                    q = self.actor(state_batch.float())
                    with torch.no_grad():
                        ''' 
                        DDQN 
                            - find argmax (indices) using actor network,
                            - find actual Q values using target-actor network.
                            - NOTE: use tensors of same shape (B, 1), otherwise unexpected broadcasting will take place.
                        '''
                        index = self.actor(state2_batch.float()).argmax(dim=1, keepdim=True)
                        q_next = self.target_actor(state2_batch.float()).gather(dim=1, index=index)

                    Y = reward_batch + self.GAMMA * ((1 - done_batch) * q_next)
                    X = q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1))

                    loss = F.smooth_l1_loss(X, Y.detach())
                    self.actor_opt.zero_grad()
                    loss.backward()
                    self.actor_opt.step()

            # reduce epsilon, print when it dies
            if self.epsilon > 0.1: self.epsilon -= 2.2/self.NUM_EPISODES 
            else:
                if not self.epsilon_checked:
                    print(f"EPSILON died on EP {ep}")
                    self.epsilon_checked = True

            # comparison of initial & final PID with score
            # print(f"PID: {saved_init_state} to {state} with score {score:.2f}")
            # print(f"score: {score:.2f}")

            # print & save score
            if mp: self.queue.put(score)
            self.scores.append(score)
            
        # save agent
        if save: 
            import datetime
            PATH = "./saved_models/" + datetime.datetime.now().strftime('%m-%d %H:%M')
            torch.save(self.actor, f"{PATH}.pth")

        self.env.close()

        import matplotlib.pyplot as plt
        plt.plot(self.scores)
        return self.scores
    
    def test_agent(self, env, MIMO=False):

        scores = []

        if not MIMO:

            for _ in range(1):
                state, _ = env.reset()
                done, score = False, 0
                print(f"Initial PID:{state}")

                while not done:
                    q = self.actor(torch.from_numpy(state).float())
                    action = np.argmax(q.detach().numpy())
                    next_state, reward, term, trunc, _ = self.env.linstep(self.action_map[action])
                    done = term or trunc
                    score += reward
                    state = next_state
                
                print(f"Final PID: {state}")
        
        elif MIMO:
            for _ in range(1):
                state, _ = env.reset(MIMO=True, online=True)
                done, score = False, 0
                print(f"Initial PID:{state}")

                while not done:
                    q = self.actor(torch.from_numpy(state).float())
                    action = np.argmax(q.detach().numpy())
                    next_state, reward, term, trunc, _ = self.env.step_online(self.action_map[action])
                    done = term or trunc
                    score += reward
                    state = next_state
                
                print(f"Final PID: {state}")

