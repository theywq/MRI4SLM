from transformers import AutoModelForCausalLM
import torch
import argparse

from models.student import MARDStudent
from models.controller import MetaController
from models.projection import Projection
from training.bilevel import bilevel_step
from utils.saliency import compute_saliency
from config import cfg

from data.dataset import load_hf_dataset
from data.tokenizer import build_tokenizer
from data.dataloader import build_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="gsm8k")
args = parser.parse_args()

teacher = AutoModelForCausalLM.from_pretrained("gpt2")
student_base = AutoModelForCausalLM.from_pretrained("gpt2")

tokenizer = build_tokenizer("gpt2")

train_ds = load_hf_dataset(args.dataset, "train")
val_ds = load_hf_dataset(args.dataset, "test")

train_loader = build_dataloader(train_ds, tokenizer)
val_loader = build_dataloader(val_ds, tokenizer)

sample_batch = next(iter(train_loader))
layers = compute_saliency(teacher, sample_batch["input_ids"])

student = MARDStudent(student_base, layers)
controller = MetaController(cfg.d_model)
proj = Projection(cfg.d_model, cfg.d_model)

opt_s = torch.optim.AdamW(student.parameters(), lr=cfg.lr_student)
opt_c = torch.optim.AdamW(controller.parameters(), lr=cfg.lr_controller)

for step, (train_batch, val_batch) in enumerate(zip(train_loader, val_loader)):
    loss_train, loss_val = bilevel_step(
        student, teacher, controller, proj,
        train_batch, val_batch, opt_s, opt_c
    )

    if step % 10 == 0:
        print(step, loss_train, loss_val)

    if step > 100:
        break