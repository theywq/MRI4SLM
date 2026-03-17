# MARD: Module-Aware Reasoning Distillation

## Overview
This repository provides a runnable PyTorch implementation of MARD, including:
- Saliency-based layer selection
- Module-aware distillation (attention / FFN)
- Bi-level optimization

## Installation
pip install -r requirements.txt

## Run Training
python train.py

## Dataset (HuggingFace)
We support loading datasets via `datasets`:
- gsm8k (math reasoning)
- math_qa

## Example
python train.py --dataset gsm8k

## Project Structure
See code for modular design (models / training / utils).
