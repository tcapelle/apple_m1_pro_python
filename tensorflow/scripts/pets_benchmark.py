import re
import os
import argparse
from time import perf_counter
from glob import glob
from typing import List
from pathlib import Path
from typing import Callable, List
from types import SimpleNamespace
from sklearn.model_selection import train_test_split

import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import layers, losses, optimizers, applications


PROJECT = "M1_TF_vs_PT"
ENTITY = "capecape"
GROUP = "tf"

config_defaults = SimpleNamespace(
    batch_size=64,
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    validation_split=0.2,
    image_size=128,
    model_name="resnet50",
    artifact_address="capecape/pytorch-M1Pro/PETS:latest",
    gpu_name="M1Pro GPU 16 Cores",
    fp16=False,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=config_defaults.image_size)
    parser.add_argument("--batch_size", type=int, default=config_defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=config_defaults.epochs)
    parser.add_argument(
        "--num_experiments", type=int, default=config_defaults.num_experiments
    )
    parser.add_argument(
        "--validation_split", type=float, default=config_defaults.validation_split
    )
    parser.add_argument(
        "--learning_rate", type=float, default=config_defaults.learning_rate
    )
    parser.add_argument("--model_name", type=str, default=config_defaults.model_name)
    parser.add_argument("--artifact_address", type=str, default=config_defaults.artifact_address)
    parser.add_argument("--gpu_name", type=str, default=config_defaults.gpu_name)
    return parser.parse_args()


AUTOTUNE = tf.data.AUTOTUNE
BACKBONE_DICT = {
    "resnet50": {
        "model": applications.ResNet50,
        "preprocess_fn": applications.resnet50.preprocess_input,
    }
}

VOCAB = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
    "american_bulldog",
    "american_pit",
    "basset_hound",
    "beagle",
    "boxer",
    "chihuahua",
    "english_cocker",
    "english_setter",
    "german_shorthaired",
    "great_pyrenees",
    "havanese",
    "japanese_chin",
    "keeshond",
    "leonberger",
    "miniature_pinscher",
    "newfoundland",
    "pomeranian",
    "pug",
    "saint_bernard",
    "samoyed",
    "scottish_terrier",
    "shiba_inu",
    "staffordshire_bull",
    "wheaten_terrier",
    "yorkshire_terrier",
]


class SamplesSec(tf.keras.callbacks.Callback):
    def __init__(self, epochs=1, batch_size=1, drop=5):
        self.epochs = epochs
        self.batch_size = batch_size
        self.drop = drop
        

    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.samples_s = 0.

    def on_epoch_begin(self, epoch, logs={}):
        self.batch_times = []

    def on_train_batch_begin(self, batch, logs={}):
        self.batch_train_start = perf_counter()

    def on_train_batch_end(self, batch, logs={}):
        t = perf_counter() - self.batch_train_start
        self.batch_times.append(t)

    def on_epoch_end(self, epoch, logs={}):
        self.batch_times.sort()
        avg_time_per_batch = sum(self.batch_times[0:-self.drop])/(len(self.batch_times)-self.drop)
        samples_s_batch = self.batch_size / avg_time_per_batch
        wandb.log({"samples_per_batch": samples_s_batch})
        self.samples_s += samples_s_batch
    
    def on_train_end(self, logs={}):
        wandb.log({"samples_per_s": self.samples_s/self.epochs})

class PetsDataLoader:
    def __init__(
        self,
        artifact_address: str,
        preprocess_fn: Callable,
        image_size: int,
        batch_size: int,
        vocab: List[str]=VOCAB,
    ):
        self.artifact_address = artifact_address
        self.dataset_path = self.get_pets()
        self.preprocess_fn = preprocess_fn
        print(self.preprocess_fn)
        self.image_size = image_size
        self.batch_size = batch_size
        self.pattern = r"(^[a-zA-Z]+_*[a-zA-Z]+)"
        self.vocab_map = {v: i for i, v in enumerate(vocab)}
        self.image_files = glob(os.path.join(self.dataset_path, "images", "*.jpg"))
        self.labels = [
            self.vocab_map[re.match(self.pattern, Path(image_file).name)[0]]
            for image_file in self.image_files
        ]

    def __len__(self):
        return len(self.image_files)

    def get_pets(self):
        api = wandb.Api()
        at = api.artifact(self.artifact_address, type="dataset")
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


def get_model(image_size: int, model_name: str, vocab: List[str]) -> Model:
    input_shape = [image_size, image_size, 3]
    input_tensor = Input(shape=input_shape)
    backbone_out = BACKBONE_DICT[model_name]["model"](
        include_top=False, input_tensor=input_tensor
    )(input_tensor)
    x = layers.GlobalAveragePooling2D()(backbone_out)
    output = layers.Dense(len(vocab))(x)
    return Model(input_tensor, output)


def train(args):
    with wandb.init(project=PROJECT, entity=ENTITY, job_type="Benchmark"):

        config = wandb.config
        config.image_size = args.image_size
        config.batch_size = args.batch_size
        config.validation_split = args.validation_split
        config.artifact_address = args.artifact_address
        config.model_name = args.model_name
        config.vocab = VOCAB
        config.epochs = args.epochs
        config.learning_rate = args.learning_rate
        config.gpu_name = args.gpu_name

        loader = PetsDataLoader(
            artifact_address=config.artifact_address,
            preprocess_fn=BACKBONE_DICT[config.model_name]["preprocess_fn"],
            image_size=config.image_size,
            batch_size=config.batch_size,
        )
        print("Dataset Size:", len(loader))

        train_dataset, val_dataset = loader.get_datasets(
            val_split=config.validation_split
        )

        model = get_model(
            image_size=config.image_size,
            model_name=config.model_name,
            vocab=VOCAB,
        )
        model.summary()

        model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizers.Adam(learning_rate=config.learning_rate),
            metrics=["accuracy"],
        )

        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=[WandbCallback(),
                       SamplesSec(config.epochs, config.batch_size)],
        )


if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train(args=args)
