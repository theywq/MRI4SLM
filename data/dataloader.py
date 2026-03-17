from torch.utils.data import DataLoader

def collate_fn(batch, tokenizer, max_len=512):
    texts = [x["text"] for x in batch]
    enc = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_len, return_tensors="pt")
    return {"input_ids": enc["input_ids"]}

def build_dataloader(dataset, tokenizer, batch_size=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=lambda x: collate_fn(x, tokenizer))