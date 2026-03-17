from transformers import AutoTokenizer

def build_tokenizer(model_name="gpt2"):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok