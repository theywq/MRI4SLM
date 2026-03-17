import torch.nn.functional as F
from config import cfg

def token_loss(s_logits, t_logits):
    p_s = F.log_softmax(s_logits / cfg.temperature, dim=-1)
    p_t = F.softmax(t_logits / cfg.temperature, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean")

def module_loss(s_h, t_h, proj=None):
    if proj is not None:
        s_h = proj(s_h)
    return F.mse_loss(s_h, t_h)