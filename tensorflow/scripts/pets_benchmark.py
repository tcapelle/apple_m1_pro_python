import re
import os
import wandb
import argparse
from glob import glob
from typing import List
from pathlib import Path
from typing import Callable
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras


PROJECT = "pytorch-M1Pro"
ENTITY = "geekyrakshit"

config_defaults = SimpleNamespace(
    batch_size=64,
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    image_size=128,
    backbone_name="resnet50",
    artifact_address="capecape/pytorch-M1Pro/PETS:latest",
    gpu_name="M1Pro GPU 16 Cores",
    fp16=False
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=config_defaults.image_size)
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
    return parser.parse_args()

AUTOTUNE = tf.data.AUTOTUNE
BACKBONE_DICT = {
    "resnet50": {
        "model": keras.applications.ResNet50,
        "preprocess_fn": keras.applications.resnet50.preprocess_input
    }
}

VOCAB = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay',
    'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 
    'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese',
    'Sphynx', 'american_bulldog', 'american_pit', 
    'basset_hound', 'beagle', 'boxer', 'chihuahua',
    'english_cocker', 'english_setter', 'german_shorthaired', 
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond',
    'leonberger', 'miniature_pinscher', 'newfoundland', 
    'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull', 
    'wheaten_terrier', 'yorkshire_terrier'
]

class PetsDataLoader:

    def __init__(
        self,
        artifact_address: str,
        preprocess_fn: Callable,
        vocab: List[str],
        image_size: int,
        batch_size: int
    ):
        self.artifact_address = artifact_address
        self.dataset_path = self.get_pets()
        self.preprocess_fn = preprocess_fn
        print(self.preprocess_fn)
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern = r'(^[a-zA-Z]+_*[a-zA-Z]+)'
        self.vocab_map = {v:i for i, v in enumerate(vocab)}
        self.image_files = glob(os.path.join(self.dataset_path, "images", "*.jpg"))
        self.labels = [
            self.vocab_map[re.match(self.pattern, Path(image_file).name)[0]]
            for image_file in self.image_files
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def get_pets(self):
        api = wandb.Api()
        at = api.artifact(self.artifact_address, type='dataset')
        dataset_path = at.download()
        return dataset_path
    
    def map_fn(self, image_file, label):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_png(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [self.image_size, self.image_size])
        image = self.preprocess_fn(image)
        return image, label
    
    def build_dataset(self, images, labels):
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.map(self.map_fn, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset
    
    def get_datasets(self, val_split: float):
        train_images, val_images, train_labels, val_labels = train_test_split(
            self.image_files, self.labels, test_size=val_split
        )
        train_dataset = self.build_dataset(train_images, train_labels)
        val_dataset = self.build_dataset(val_images, val_labels)
        return train_dataset, val_dataset