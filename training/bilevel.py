from training.losses import token_loss, module_loss
from config import cfg

def bilevel_step(student, teacher, controller, proj,
                 train_batch, val_batch, opt_s, opt_c):

    input_ids = train_batch["input_ids"]

    with torch.no_grad():
        t_logits, t_hidden = teacher(input_ids, output_hidden_states=True)

    s_logits, s_hidden = student(input_ids)

    loss_tok = token_loss(s_logits, t_logits)
    loss_mod = 0

    for l in student.layers:
        h_t = t_hidden[l]
        h_s = s_hidden[l]

        w_attn, w_ffn = controller(h_t)

        loss_mod += (w_attn.mean() * module_loss(h_s, h_t, proj) +
                     w_ffn.mean() * module_loss(h_s, h_t, proj))

    loss_train = cfg.lambda_token * loss_tok + loss_mod

    opt_s.zero_grad()
    loss_train.backward()
    opt_s.step()

    # controller update
    input_ids_val = val_batch["input_ids"]

    with torch.no_grad():
        t_logits_val, _ = teacher(input_ids_val, output_hidden_states=True)

    s_logits_val, _ = student(input_ids_val)
    loss_val = token_loss(s_logits_val, t_logits_val)

    opt_c.zero_grad()
    loss_val.backward()
    opt_c.step()

    return loss_train.item(), loss_val.item()