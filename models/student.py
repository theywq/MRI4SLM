import torch.nn as nn
from .adapter import Adapter
from config import cfg

class MARDStudent(nn.Module):
    def __init__(self, base_model, layers):
        super().__init__()
        self.model = base_model
        self.layers = layers

        self.attn_adapters = nn.ModuleDict({
            str(l): Adapter(cfg.d_model, cfg.adapter_rank) for l in layers
        })
        self.ffn_adapters = nn.ModuleDict({
            str(l): Adapter(cfg.d_model, cfg.adapter_rank) for l in layers
        })

    def forward(self, input_ids):
        outputs = self.model(input_ids, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)

        for l in self.layers:
            h = hidden_states[l]
            hidden_states[l] = h + \
                self.attn_adapters[str(l)](h) + \
                self.ffn_adapters[str(l)](h)

        return outputs.logits, hidden_states