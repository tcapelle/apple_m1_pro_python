## Author: Thomas Capelle, Soumik Rakshit
## Mail:   tcapelle@wandb.com, soumik.rakshit@wandb.com

""""Benchmarking apple M1Pro with Tensorflow
@wandbcode{apple_m1_pro}"""

import re, math, argparse
from types import SimpleNamespace
from pathlib import Path
from time import perf_counter

import wandb
from PIL import Image
from tqdm import tqdm


import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.cuda.amp import autocast

from utils import get_gpu_name
from pets import get_pets_dataloader

PROJECT = "pt2"
ENTITY = "capecape"
GROUP = "pytorch"

config_defaults = SimpleNamespace(
    batch_size=128,
    device="cuda",
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    image_size=512,
    model_name="resnet50",
    dataset="PETS",
    num_workers=0,
    gpu_name=get_gpu_name(),
    mixed_precision=False,
    channels_last=False,
    optimizer="Adam",
    compile=False,
    tags=None,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=ENTITY)
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--image_size', type=int, default=config_defaults.image_size)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    parser.add_argument('--device', type=str, default=config_defaults.device)
    parser.add_argument('--gpu_name', type=str, default=config_defaults.gpu_name)
    parser.add_argument('--num_workers', type=int, default=config_defaults.num_workers)
    parser.add_argument('--mixed_precision', action="store_true")
    parser.add_argument('--channels_last', action="store_true")
    parser.add_argument('--optimizer', type=str, default=config_defaults.optimizer)
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--tags', type=str, default=None)
    return parser.parse_args()


def get_model(n_out, arch="resnet50"):
    model = getattr(tv.models, arch)(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_out)
    return model


def check_cuda(config):
    if torch.cuda.is_available():
        config.device = "cuda"
        config.mixed_precision = True
    return config

def train(config):
    config = check_cuda(config)
    tags = [f"pt{torch.__version__}", f"cuda{torch.version.cuda}"] + (config.tags.split(",") if config.tags is not None else [])
    print(tags)
    with wandb.init(project=PROJECT, entity=args.entity, group=GROUP, tags=tags, config=config):

        # Copy your config 
        config = wandb.config

        # Get the data
        train_dl = get_pets_dataloader(batch_size=config.batch_size, 
                                       image_size=config.image_size,
                                       num_workers=config.num_workers)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        model = get_model(len(train_dl.dataset.vocab), config.model_name)
        model.to(config.device)
        if config.channels_last:
            model.to(memory_format=torch.channels_last)
        if torch.__version__ >= "2.0" and config.compile:
            print("Compiling model...")
            model = torch.compile(model)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = getattr(torch.optim, config.optimizer)
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)

       # Training
        example_ct = 0
        for epoch in tqdm(range(config.epochs)):
            tf = t0 = perf_counter()
            model.train()
            for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
                images, labels = images.to(config.device), labels.to(config.device)
                if config.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                
                # compute the model froward and backward pass time
                ti = perf_counter()
                if config.mixed_precision:
                    with autocast():
                        outputs = model(images)
                        train_loss = loss_func(outputs, labels)
                else:
                    outputs = model(images)
                    train_loss = loss_func(outputs, labels)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tf_with_dataloader = perf_counter() - tf
                tf = perf_counter()


                # log the metrics
                example_ct += len(images)
                metrics = {"train/train_loss": train_loss, 
                           "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                           "train/example_ct": example_ct,
                           "samples_per_sec":len(images)/(tf-ti),
                           "samples_per_sec_dl":len(images)/tf_with_dataloader,
                           "samples_per_sec_epoch":example_ct/(tf-t0)}

                if step + 1 < n_steps_per_epoch:
                    # 🐝 Log train metrics to wandb 
                    wandb.log(metrics)

if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train(config=args)
