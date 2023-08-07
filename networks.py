import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 150)
        self.layer3 = nn.Linear(150, 100)
        self.layer4 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = self.layer4(x)

        return x

class duel_model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        # common layer(s)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 150),
            nn.LeakyReLU(),
        )

        # advantage layer(s)
        self.adv_layer = nn.Sequential(
            nn.Linear(150, 100),
            nn.LeakyReLU(),
            nn.Linear(100, output_dim),
        )

        # value layer
        self.val_layer = nn.Sequential(
            nn.Linear(150, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        
        a = self.adv_layer(x)
        v = self.val_layer(x)

        q = v + a - a.mean(dim=-1, keepdim=True)

        return q