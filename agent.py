from networks import *
from tqdm import tqdm
from collections import deque
import numpy as np, copy, random

class Agent:
    def __init__(self, env):
        self.GAMMA = 0.9
        self.NUM_EPISODES = 1_500
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

            if ep==1: raise Exception

            while not done:
                
                synch_i += 1
                if synch_i % self.SYNCH_FREQ == 0:
                    self.target_actor.load_state_dict(self.actor.state_dict())

                action = self.get_action(state)
                
                next_state, reward, term, trunc, _ = self.env.step_online(np.array(self.action_map[action]))
                done = term or trunc
                exp = (torch.tensor(state), action, reward, torch.tensor(next_state), done)
                self.replay.append(exp)
                score += reward

                print(f"------on episode {ep}, time {synch_i}------")
                print(f"initial PID: {state}")
                print(f"Action: {self.action_map[action]}")
                print(f"reward: {reward}")
                print(f"next PID {next_state}")

                state = next_state

                if len(self.replay) > self.BATCH_SIZE:
                    minibatch = random.sample(self.replay, self.BATCH_SIZE)
                    state_batch = torch.stack([s1 for s1, a, r, s2, d in minibatch])
                    action_batch = torch.Tensor([a for s1, a, r, s2, d in minibatch])
                    reward_batch = torch.Tensor([r for s1, a, r, s2, d in minibatch])
                    state2_batch = torch.stack([s2 for s1, a, r, s2, d in minibatch])
                    done_batch = torch.Tensor([d for s1, a, r, s2, d in minibatch])

                    q = self.actor(state_batch.float())
                    with torch.no_grad():
                        q_next = self.target_actor(state2_batch.float())
                    
                    Y = reward_batch + self.GAMMA * ((1 - done_batch) * torch.max(q_next, dim=1)[0])
                    X = q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()

                    loss = F.mse_loss(X, Y.detach())
                    self.actor_opt.zero_grad()
                    loss.backward()
                    self.actor_opt.step()

            # reduce epsilon, print when it dies
            if self.epsilon > 0.1: self.epsilon -= 2/self.NUM_EPISODES 
            else:
                if not self.epsilon_checked:
                    print(f"EPSILON died on EP {ep}")
                    self.epsilon_checked = True

            # comparison of initial & final PID with score
            # print(f"PID: {saved_init_state} to {state} with score {score:.2f}")

            # print & save score
            if mp: self.queue.put(score)
            self.scores.append(score)
            
        # save agent
        if save: 
            import datetime
            PATH = "./saved_models/" + datetime.datetime.now().strftime('%m-%d %H:%M')
            torch.save(self.actor, f"actor-{PATH}.pth")

        self.env.close()
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
                state, _ = env.reset(MIMO=True)
                done, score = False, 0
                print(f"Initial PID:{state}")

                while not done:
                    q = self.actor(torch.from_numpy(state).float())
                    action = np.argmax(q.detach().numpy())
                    next_state, reward, term, trunc, _ = self.env.linstep_MIMO(self.action_map[action])
                    done = term or trunc
                    score += reward
                    state = next_state
                
                print(f"Final PID: {state}")

