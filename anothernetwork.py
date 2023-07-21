import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, output_dim*3)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))

        x = x.reshape(-1, 3, 3).squeeze()
        x = F.softmax(x, dim=0)

        return x