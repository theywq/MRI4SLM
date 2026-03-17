import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, d_model, rank):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.up(self.act(self.down(x)))