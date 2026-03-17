from datasets import load_dataset

def load_hf_dataset(name="gsm8k", split="train"):
    if name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split)
        ds = ds.map(lambda x: {"text": x["question"] + "\\n" + x["answer"]})
    elif name == "math_qa":
        ds = load_dataset("math_qa", split=split)
        ds = ds.map(lambda x: {"text": x["Problem"]})
    else:
        raise ValueError(f"Unknown dataset {name}")
    return ds