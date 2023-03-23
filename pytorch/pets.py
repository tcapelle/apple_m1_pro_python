import re

from pathlib import Path

import wandb


from PIL import Image

import torch
import torchvision.transforms as T


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



def get_pets_dataloader(batch_size, image_size=224, num_workers=0, **kwargs):
    "Get a training dataloader"
    dataset_path = get_pets()
    ds = Pets(dataset_path, image_size=image_size)
    loader = torch.utils.data.DataLoader(ds, 
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         **kwargs)
    return loader

class OneBatchDataLoader:
    def __init__(self, dl, N=10):
        self.dl = dl
        self.batch = next(iter(dl))
        self.N = N
    
    def __iter__(self):
        yield self.batch

    def __len__(self):
        return self.N


def get_fast_pets_dataloader(batch_size, image_size=224, num_workers=0, **kwargs):
    "Get a training dataloader"
    dl = get_pet_dataloader(batch_size, image_size=image_size, num_workers=num_workers, **kwargs)
    return OneBatchDataLoader(dl)