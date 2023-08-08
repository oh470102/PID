from networks import *
from tqdm import tqdm
from collections import deque
import numpy as np, copy, random

class Agent:
    '''
    The reinforcement learning (RL) agent, that learns to find the best PID coefficients.
    
    RL Algorithm:
        - uses Deep Q-Network along with
        - experience replay
        - target network
        - double Q-learning
        - dueling network
        - noisy network
        
    Description of the environment
        - State: the current PID coefficients
        - Action: the change in each PID coefficients
        - Reward: improvement in stability (offline) or ISE (online)
        - Done: environment terminates after 50 time-steps

    Training Process:
        - Offline Tuning
            - In each episode, agent starts with randomly initialized PID coefficients
            - In each time-step of the episode, the agent may change each coefficient by -1, 0, or 1.
            - After each variation of the coefficients, a reward is given based on the stability improvement.
            - The most stable PID coefficient found is kept track of, and is later used as the baseline model.
        - Online Tuning
            - In each episode, the agent starts with the baseline PID found previously, but with some noise (-5 ~ +5).
            - In each time-step of the episode, the agent may change each coefficient by -1, 0, or 1, but a Gaussian noise N~(0, 0.5) is added.
            - A reward is given based on the improvement in ISE (the smaller the better), and whether the controller remained stable enough.
            - The best-performing PID coefficient (still ought to be stable) is kept track of, which is the final model.
        
    '''
    def __init__(self, env):

        # Epochs
        self.NUM_EPISODES = 1000

        # Hyperparameters
        self.GAMMA = 0.99
        self.LEARNING_RATE = 1e-3

        # Experience replay, target network stuff
        self.BATCH_SIZE = 200
        self.MEM_SIZE = 100_000
        self.replay = deque(maxlen=self.MEM_SIZE)
        self.SYNCH_FREQ = 500
        self.synch_i = 0

        # RL environment stuff
        self.env = env
        self.state_dim = 6
        self.action_dim = 729 # 3**6
        self.action_map = [[a,b,c,d,e,f] for a in [1, -1, 0] for b in [1, -1, 0] for c in [1, -1, 0] for d in [1, -1, 0] for e in [1, -1, 0] for f in [1, -1, 0]]

        # Neural network stuff
        self.actor = Noisymodel(self.state_dim, self.action_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.LEARNING_RATE)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Epsilon-greddy stuff
        self.epsilon = 1.0
        self.epsilon_checked = False
        
        # Miscellaneous
        self.scores = []

    def update_target_network(self) -> None:

        '''
        Manages target network synch frequency. 
        Executes an hard update if time to do so.
        May change to soft-updates later.
        '''
        
        self.synch_i += 1
        if self.synch_i % self.SYNCH_FREQ == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())

        return 

    def get_action(self, state: np.ndarray, epsilon: bool = False) -> int : 

        '''
        returns action index from state 
            - action index is an int ~ [0, action_dim - 1]
            - may or may not use epsilon greedy 
        '''

        # feed state into network, get Q-values
        q = self.actor(torch.from_numpy(state).float())
        q_ = q.detach().numpy()

        # if using e-greedy
        if epsilon:

            if random.random() < self.epsilon:
                action = np.random.randint(0, int(self.action_dim))

            else:
                action = np.argmax(q_)

        # if not, noise is inherent in noise-net
        else:
            action = np.argmax(q_)

        return action
    
    def learn(self) -> None:

        '''
        Performs neural network optimization from a batch of 'experiences' from the replay buffer. 
        '''

        # sample minibatch from replay buffer
        # NOTE: make sure all batches are 2D tensors, otherwise unexpected broadcasting will take place.
        minibatch = random.sample(self.replay, self.BATCH_SIZE)
        state_batch = torch.stack([s1 for s1, a, r, s2, d in minibatch])
        action_batch = torch.Tensor([a for s1, a, r, s2, d in minibatch])
        reward_batch = torch.Tensor([r for s1, a, r, s2, d in minibatch]).reshape(-1, 1)
        state2_batch = torch.stack([s2 for s1, a, r, s2, d in minibatch])
        done_batch = torch.Tensor([d for s1, a, r, s2, d in minibatch]).reshape(-1, 1)

        q = self.actor(state_batch.float())
        with torch.no_grad():
            ''' 
            Double DQN 
                - find argmax (indices) using actor network,
                - find actual Q values using target-actor network.
            '''
            index = self.actor(state2_batch.float()).argmax(dim=1, keepdim=True)
            q_next = self.target_actor(state2_batch.float()).gather(dim=1, index=index)

        # Calculate TD-Target
        Y = reward_batch + self.GAMMA * ((1 - done_batch) * q_next)

        # Get Q value outputs from our network
        X = q.gather(dim=1, index=action_batch.long().unsqueeze(dim=1))

        # perform gradient descent, clipping grad
        loss = F.smooth_l1_loss(X, Y.detach())
        self.actor_opt.zero_grad()
        loss.backward()
        '''
        Duel DQN
            - clips gradient to prevent gradient explosions
        '''
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # reset noise for NoiseNet for both actor and target-actor networks.
        # NOTE: only do this if e-greedy is not used.
        self.actor.reset_noise()
        self.target_actor.reset_noise()

    def reduce_epsilon(self, ep: int) -> None:

        '''
        Decay epsilon (e-greedy)
            - lower bound = 0.1.
            - does not fall below the lower bound.
            - reaches lower bound when the training is 40% done.
            - when it dies (i.e. reaches lower bound), a line is printed to notify
        '''

        if self.epsilon > 0.1: 
            self.epsilon -= 2.2/self.NUM_EPISODES 

        else:
            if not self.epsilon_checked:
                print(f"EPSILON died on EP {ep}")
                self.epsilon_checked = True

        return 
    
    def log(self, exp: tuple) -> None:

        '''
        A logging function that prints out each transition
            - enable to keep track of any logic errors
            - 'Transition' must be a tuple (s, a, r, s', d)
        '''

        state, action, reward, next_state, _ = exp

        if self.synch_i % 10 == 0:
            print(f"initial PID: {state}")
            print(f"Action: {self.action_map[action]}")
            print(f"reward: {reward}")
            print(f"next PID: {next_state}")


    def train(self, save: bool = False) -> np.ndarray:

        '''
        Trains the agent via DQN
        '''

        for ep in tqdm(range(self.NUM_EPISODES)):

            # reset and retrieve initial PID
            state, _ = self.env.reset(MIMO=True, online=True) 
            done, score = False, 0

            # episode main loop
            while not done:
                
                # update target network 
                self.update_target_network()

                # get / process / add noise / clip action
                action = self.get_action(state)
                action_ = self.action_map[action] + np.random.normal(loc=0, scale=0.5, size=6)
                action_ = action_.clip(-2, 2)

                # take step in environment
                next_state, reward, term, trunc, _ = self.env.step_online(action_)
                done = term or trunc

                # save experience in buffer
                exp = (torch.tensor(state), action, reward, torch.tensor(next_state), done)
                self.replay.append(exp)
                score += reward
                state = next_state

                # logging stuff (NOTE: temporarily commented out)
                # self.log(exp)

                # learn from buffer 
                if len(self.replay) > self.BATCH_SIZE:
                    self.learn()

            # decay epsilon if using e-greedy
            self.reduce_epsilon(ep=ep)

            # record score
            self.scores.append(score)
            
        # save agent
        if save: 
            import datetime
            PATH = "./saved_models/" + datetime.datetime.now().strftime('%m-%d %H:%M')
            torch.save(self.actor, f"{PATH}.pth")

        # close env, return all scores
        self.env.close()
        return self.scores
    
