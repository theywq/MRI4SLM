from config import cfg

def compute_saliency(teacher, input_ids):
    outputs = teacher(input_ids, output_hidden_states=True)

    for h in outputs.hidden_states:
        h.retain_grad()

    loss = outputs.logits.mean()
    loss.backward()

    scores = [h.grad.abs().mean().item() for h in outputs.hidden_states]
    topk = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:cfg.topk_layers]
    return topk