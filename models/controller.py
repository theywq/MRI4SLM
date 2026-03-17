import torch.nn as nn
import torch.nn.functional as F

class MetaController(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, h):
        pooled = h.mean(dim=1)
        w = F.softmax(self.net(pooled), dim=-1)
        return w[:, 0], w[:, 1]