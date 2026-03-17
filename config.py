class Config:
    d_model = 768
    adapter_rank = 16
    lr_student = 2e-4
    lr_controller = 5e-5
    weight_decay = 0.01
    lambda_token = 1.0
    lambda_attn = 0.5
    lambda_ffn = 0.5
    temperature = 1.0
    topk_layers = 4

cfg = Config()