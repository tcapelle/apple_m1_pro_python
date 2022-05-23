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

PROJECT = "M1_TF_vs_PT"
ENTITY = "capecape"
GROUP = "pytorch"

config_defaults = SimpleNamespace(
    batch_size=64,
    device="mps",
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    image_size=128,
    model_name="resnet50",
    dataset="PETS",
    num_workers=0,
    gpu_name="M1Pro GPU 16 Cores",
    fp16=False,
    optimizer="Adam"
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
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--optimizer', type=str, default=config_defaults.optimizer)
    return parser.parse_args()

def get_pets():
    api = wandb.Api()
    at = api.artifact('capecape/pytorch-M1Pro/PETS:v1', type='dataset')
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


    def __init__(self, pets_path, image_size=224):
        self.path = Path(pets_path)
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
                                         num_workers=num_workers,
                                         **kwargs)
    return loader


def get_model(n_out, arch="resnet18", pretrained=True):
    model = getattr(tv.models, arch)(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, n_out)
    return model


def train(config=config_defaults):
    config.device = "cuda" if torch.cuda.is_available() else config.device
    config.fp16 = config.device=="cuda" if config.fp16 else config.fp16

    with wandb.init(project=PROJECT, entity=args.entity, group=GROUP, config=config):

        # Copy your config 
        config = wandb.config

        # Get the data
        train_dl = get_dataloader(dataset_path=get_pets(), 
                                  batch_size=config.batch_size, 
                                  image_size=config.image_size,
                                  num_workers=config.num_workers)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

        model = get_model(len(train_dl.dataset.vocab), config.model_name)
        model.to(config.device)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = getattr(torch.optim, config.optimizer)
        optimizer = optimizer(model.parameters(), lr=config.learning_rate)

       # Training
        example_ct = 0
        step_ct = 0
        for epoch in tqdm(range(config.epochs)):
            model.train()
            for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
                images, labels = images.to(config.device), labels.to(config.device)

                ti = perf_counter()
                if config.fp16:
                    with autocast():
                        outputs = model(images)
                        train_loss = loss_func(outputs, labels)
                else:
                    outputs = model(images)
                    train_loss = loss_func(outputs, labels)
                train_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tf = perf_counter()
                example_ct += len(images)
                metrics = {"train/train_loss": train_loss, 
                           "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                           "train/example_ct": example_ct,
                           "samples_per_sec":len(images)/(tf-ti)}

                if step + 1 < n_steps_per_epoch:
                    # 🐝 Log train metrics to wandb 
                    wandb.log(metrics)

                step_ct += 1
                
                
if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train(config=args)
