## Author: Thomas Capelle, Soumik Rakshit
## Mail:   tcapelle@wandb.com, soumik.rakshit@wandb.com

""""Benchmarking apple M1Pro with Tensorflow
@wandbcode{apple_m1_pro}"""

import re, math, argparse
from types import SimpleNamespace
from pathlib import Path
from time import perf_counter
import contextlib

import wandb
from PIL import Image
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch import autocast

from utils import get_gpu_name


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

def get_pets(version="v3"):
    api = wandb.Api()
    at = api.artifact(f'capecape/pytorch-M1Pro/PETS:{version}', type='dataset')
    dataset_path = at.download()
    return dataset_path

class Pets(torch.utils.data.Dataset):
    pat = r'(^[a-zA-Z]+_*[a-zA-Z]+)'
    vocab = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 
             'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit', 
             'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker', 'english_setter', 'german_shorthaired', 
             'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 
             'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull', 
             'wheaten_terrier', 'yorkshire_terrier']
    vocab_map = {v:i for i,v in enumerate(vocab)}


    def __init__(self, pets_path="artifacts/PETS:v3", image_size=224):
        self.path = Path(pets_path)
        print(f"Dataset path: {self.path}")
        self.files = list(self.path.glob("images/*.jpg"))
        self.tfms =T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self.vocab_map = {v:i for i, v in enumerate(self.vocab)}
    
    @staticmethod
    def load_image(fn, mode="RGB"):
        "Open and load a `PIL.Image` and convert to `mode`"
        im = Image.open(fn)
        im.load()
        im = im._new(im.im)
        return im.convert(mode) if mode else im
    
    def __getitem__(self, idx):
        file = self.files[idx]
        return self.tfms(self.load_image(str(file))), self.vocab_map[re.match(self.pat, file.name)[0]]
        
    def __len__(self): return len(self.files)


def get_dataloader(dataset_path, batch_size, image_size=224, num_workers=0, **kwargs):
    "Get a training dataloader"
    ds = Pets(dataset_path, image_size=image_size)
    loader = torch.utils.data.DataLoader(ds, 
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         multiprocessing_context='spawn',
                                         num_workers=num_workers,
                                         **kwargs)
    return loader


def get_model(n_out, arch="resnet50"):
    model = getattr(tv.models, arch)(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_out)
    return model


def check_cuda(config):
    if torch.cuda.is_available():
        config.device = "cuda"
        config.mixed_precision = True
    return config

@contextlib.contextmanager
def timeit(s="func_name"):
    start = perf_counter()
    yield
    end = perf_counter()
    print(f"Elapsed in {s}: {end-start:.2f}s")


def train(config):
    config = check_cuda(config)
    tags = [f"pt{torch.__version__}", f"cuda{torch.version.cuda}"] + (config.tags.split(",") if config.tags is not None else [])
    print(tags)

    # Get the data
    with timeit("get_dataloader"):
        train_dl = get_dataloader(dataset_path="artifacts/PETS:v3", 
                                    batch_size=config.batch_size, 
                                    image_size=config.image_size,
                                    num_workers=config.num_workers)
    with timeit("get_model"):
        model = get_model(len(train_dl.dataset.vocab), config.model_name)
        model.to(config.device)
        model.train()
        if config.channels_last:
            model.to(memory_format=torch.channels_last)
        with timeit("compile"):
            if torch.__version__ >= "2.0" and config.compile:
                print("Compiling model...")
                model = torch.compile(model)

    # Make the loss and optimizer
    with timeit("loss"):
        loss_func = nn.CrossEntropyLoss()
        optimizer = getattr(torch.optim, config.optimizer)
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)

    # Training
    example_ct = 0
    samples_per_sec = 0.
    times_per_batch = []
    for epoch in tqdm(range(config.epochs)):
        t0 = perf_counter()
        for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
            images, labels = images.to(config.device), labels.to(config.device)
            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            
            # compute the model froward and backward pass time
            ti = perf_counter()
            if config.mixed_precision:
                with autocast("cuda"):
                    outputs = model(images)
                    train_loss = loss_func(outputs, labels)
            else:
                outputs = model(images)
                train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tf = perf_counter()
            times_per_batch.append(tf-ti)
            # log the metrics
            example_ct += len(images)
        print(f"\nEpoch {epoch} took {perf_counter() - t0:.2f}s")
        print(f"Average samples/sec: {example_ct / (perf_counter() - t0):.2f}")
        print(f"Average batch time: {np.mean(times_per_batch):.2f}s")
                
if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train(config=args)
