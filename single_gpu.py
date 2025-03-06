import torch

from functools import cached_property
from typing import Literal
from transformers import AutoModelForCausalLM
from base_exp import BaseExp


class ForwardSingleGPU(BaseExp):
    model_path: str = "Qwen/Qwen2.5-7B"
    dtype: str = "bfloat16"
    max_len: int = 512

    def run(self, init_method: Literal["meta_accelerate", "meta_torch", "cuda"]):
        from tqdm import tqdm

        model = self.model(init_method).cuda()
        model.gradient_checkpointing_enable()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

        pbar = tqdm(self.dataset, desc="Training on sample_sft with single GPU")
        losses = []
        for item in pbar:
            input_ids = torch.tensor(item[:self.max_len], device="cuda:0").unsqueeze(0)
            loss = self.train_step(input_ids, model, optimizer)
            pbar.set_postfix(loss=loss.item())
            losses.append(loss.item())

        print(f"{init_method} mean loss", sum(losses) / len(losses))
        
    def train_step(self, input_ids, model, optimizer):
        logits = model(input_ids).logits
        loss = self.language_loss(logits, input_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss

    def model(self, init_method: Literal["meta_accelerate", "meta_torch", "cuda"]):
        if init_method == "meta_accelerate":
            return self.model_meta_accelerate
        elif init_method == "meta_torch":
            return self.model_meta_torch
        elif init_method == "cuda":
            return self.model_cuda
        else:
            raise ValueError(f"Invalid init_method: {init_method}")
        
    @cached_property
    def model_meta_accelerate(self):
        from accelerate import init_empty_weights
        import warnings
        
        warnings.filterwarnings("ignore", category=UserWarning)
        
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )
        model = model.to_empty(device="cuda:0")
        
        another_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        )
        model.load_state_dict(another_model.state_dict())
        
        return model

    @cached_property
    def model_meta_torch(self):
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2",
            )
        model = model.to_empty(device="cuda:0")
        
        another_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
        )
        model.load_state_dict(another_model.state_dict())
        
        return model
    
    @cached_property
    def model_cuda(self):
        return AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
        ).cuda()
    
    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer
        
        return AutoTokenizer.from_pretrained(self.model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        dest="init_method",
        type=str,
        default="meta_accelerate",
        choices=["meta_accelerate", "meta_torch", "cuda"],
    )
    args = parser.parse_args()

    forward_single_gpu = ForwardSingleGPU()
    forward_single_gpu.run(args.init_method)
