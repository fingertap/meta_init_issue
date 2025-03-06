import torch

from functools import cached_property


class SftDataset:
    def __init__(self, data_items: list[dict], tokenizer):
        self.tokenizer = tokenizer
        self.messages = [tokenizer.apply_chat_template(x) for x in data_items]
        
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index: int):
        return self.messages[index]


class BaseExp:
    @cached_property
    def dataset(self):
        import json
        from pathlib import Path
        
        data_path = Path(__file__).parent / "data/sample_sft_data.json"
        with open(data_path, "r") as f:
            data_items = json.load(f)
        return SftDataset(data_items, self.tokenizer)
    
    def language_loss(self, logits: torch.Tensor, labels, ignore_index: int = -100):
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(logits[:-1], labels[1:], ignore_index=ignore_index)
        return loss

    @property
    def tokenizer(self):
        """This is for dataset construction."""
