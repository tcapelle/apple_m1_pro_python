import math, re, subprocess
from sys import platform
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

def get_apple_hardware():
    "Get apple hardware info"
    cpu_info = subprocess.run(["system_profiler","SPHardwareDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    gpu_info = subprocess.run(["system_profiler","SPDisplaysDataType"], stdout=subprocess.PIPE).stdout.decode("utf-8") 
    system_info = dict(
        cpu = re.search(r'Chip:\s+(.+)', cpu_info).group(1),
        cpu_cores = re.search(r'Number of Cores:\s+(\d+)', cpu_info).group(1),
        memory = re.search(r'Memory:\s+(\d+)\s+GB', cpu_info).group(1),
        gpu = re.search(r'Chipset Model:\s+(.+)', gpu_info).group(1),
        gpu_cores = re.search(r'Total Number of Cores:\s+(\d+)', gpu_info).group(1),
        )
    return system_info

def get_gpu_name():
    if platform == "darwin":
        system_info = get_apple_hardware()
        return f"{system_info['gpu']} GPU {system_info['gpu_cores']} Cores"
    elif torch.cuda.is_available():
        return torch.cuda.get_device_name()
    return None

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
    mixed_precision: bool=False
    syncro: bool=False

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config_defaults.learning_rate)
        self.n_steps_per_epoch = math.ceil(len(self.train_dl.dataset) / self.train_dl.batch_size)
        
    def do_one_batch(self, batch):
        batch = to_device(batch, device=self.device)
        if self.mixed_precision:
            with autocast():
                outputs = self.model(**batch)
                loss = outputs.loss
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        return loss

    def do_one_epoch(self, dl, epoch):
        tf=perf_counter()
        for step, batch in enumerate(tqdm(dl, leave=False)):
            ti = perf_counter()
            self.optimizer.zero_grad()
            loss = self.do_one_batch(batch)
            loss.backward()
            self.optimizer.step()
            if self.syncro:
                torch.cuda.synchronize(device="cuda")
            tf_with_dataloader = perf_counter() - tf
            tf = perf_counter()
            self.example_ct += len(batch["labels"])
            metrics = {"train/train_loss": loss, 
                        "train/epoch": (step + 1 + (self.n_steps_per_epoch * epoch)) / self.n_steps_per_epoch, 
                        "train/example_ct": self.example_ct,
                        "seq_per_sec":len(batch["labels"])/(tf-ti),
                        "seq_per_sec_dl":len(batch["labels"])/tf_with_dataloader,}
            if step + 1 < self.n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)

            self.step_ct += 1

    def fit(self, epochs):
        self.example_ct = 0
        self.step_ct = 0
        self.model.train()
        for epoch in tqdm(range(epochs)):
            self.do_one_epoch(self.train_dl, epoch)
    
    def inference(self, dl, repeat=10):
        self.model.eval()
        batch = next(iter(dl))
        N = len(batch["labels"])
        inference_times = []
        for _ in tqdm(range(repeat)):
            with torch.no_grad():
                ti = perf_counter()
                _ = self.do_one_batch(batch)
                tf = perf_counter()
                inference_times.append(N/(tf-ti))
        wandb.log({"inference_seq_per_sec": sum(inference_times)/repeat})