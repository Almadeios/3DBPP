import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=128):
        super(PointNet, self).__init__()
        self.mlp1 = nn.Linear(input_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, output_dim)

    def forward(self, x):
        assert x.shape[2] == 3, f"Expected shape [B, N, 3], got {x.shape}"
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        x = self.mlp3(x)
        x = torch.max(x, dim=1)[0]
        return x

