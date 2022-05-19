import math
from types import SimpleNamespace
from dataclasses import dataclass
from time import perf_counter

import wandb
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

config_defaults = SimpleNamespace(
    learning_rate=1e-3,
)

def to_device(batch, device="cpu"):
    "Move tensors to device"
    if isinstance(batch, torch.Tensor):
        batch.to(device)
    elif isinstance(batch, dict):
        for k,v in batch.items():
            batch[k] = v.to(device)
    else:
        raise Exception(f"Can't put your batch of type {type(batch)} into device: {device}")
    return batch

@dataclass
class MicroTrainer:
    model: torch.nn.Module
    train_dl: DataLoader
    test_dl: DataLoader=None
    device: str="mps"
    fp16: bool=False

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config_defaults.learning_rate)
        self.n_steps_per_epoch = math.ceil(len(self.train_dl.dataset) / self.train_dl.batch_size)
        
    def do_one_batch(self, batch):
        batch = to_device(batch, device=self.device)
        if self.fp16:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        return loss

    def do_one_epoch(self, dl, epoch):
        for step, batch in enumerate(tqdm(dl, leave=False)):
            ti = perf_counter()
            self.optimizer.zero_grad()
            loss = self.do_one_batch(batch)
            loss.backward()
            self.optimizer.step()
            tf = perf_counter()


            self.example_ct += len(batch["labels"])
            metrics = {"train/train_loss": loss, 
                        "train/epoch": (step + 1 + (self.n_steps_per_epoch * epoch)) / self.n_steps_per_epoch, 
                        "train/example_ct": self.example_ct,
                        "sqe_per_sec":len(batch["labels"])/(tf-ti)}
            if step + 1 < self.n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)

            self.step_ct += 1

    def fit(self, epochs):
        self.example_ct = 0
        self.step_ct = 0
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.do_one_epoch(self.train_dl, epoch)
    
    def inference(self, repeat=10):
        self.model.eval()
        batch = next(iter(self.train_dl))
        N = len(batch["labels"])
        inference_times = []
        for _ in tqdm(range(repeat)):
            with torch.inference_mode():
                ti = perf_counter()
                _ = self.do_one_batch(batch)
                tf = perf_counter()
                inference_times.append(N/(tf-ti))
        wandb.log({"inference_seq_per_sec": sum(inference_times)/repeat})