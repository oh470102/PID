from networks import *
from agent import *
from collections import deque
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

class mpAgent(Agent):
    def __init__(self, env):
        super().__init__(env)

        self.actor = PolicyNetwork(self.state_dim, self.action_dim)
        self.actor.share_memory()
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
        
        self.value = ValueNetwork(self.state_dim)
        self.value.share_memory()
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=3e-3)

        self.traj = deque(maxlen=10000)
        self.scores = []

        self.processes = []
        self.queue = mp.Queue()
        self.PIDqueue = mp.Queue()
        self.n_workers = 5

    def mp_train(self):

        for i in range(self.n_workers):
            p = mp.Process(target=self.train, args=(True,))
            p.start()
            self.processes.append(p)

        for p in self.processes:
            p.join()

        scores = []
        while not self.queue.empty():
            scores.append(self.queue.get())

        best_PID = []
        while not self.PIDqueue.empty():
            best_PID.append(self.PIDqueue.get())

        self.graph_results(scores)

        print(best_PID)

    def graph_results(self, scores):

        # plot score graph
        plt.plot(scores)
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.grid()
        plt.show()

        # # plot P 
        # colors = np.random.rand(50)
        # plt.scatter([i for i in range(len(self.P))], self.P, c=colors, alpha=0.5, cmap='Spectral')
        # plt.colorbar()
        # plt.show()

        # # plot I 
        # colors = np.random.rand(50)
        # plt.scatter([i for i in range(len(self.I))], self.I, c=colors, alpha=0.5, cmap='Spectral')
        # plt.colorbar()
        # plt.show()

        # # plot D
        # colors = np.random.rand(50)
        # plt.scatter([i for i in range(len(self.D))], self.D, c=colors, alpha=0.5, cmap='Spectral')
        # plt.colorbar()
        # plt.show()