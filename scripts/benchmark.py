import wandb
from fastcore.script import *
import tensorflow_datasets as tfds
from keras_cvp import train

# globals you should not vary...
N_CLASSES = 10
DATASET = "cifar10"
IMG_SIZE = 128
BASE_MODEL = "MobileNetV2"

# params...
PROJECT = "m1-benchmark"
TEAM    = "m1-benchmark"  #I know, same name
MODELS = ["MobileNetV2", "ResNet50"]
HW_NAME = 'M1Pro_14inch_GPU16'
BS = 128
EPOCHS = 4
REPEAT = 3


def run_one(hw_name = HW_NAME, 
            trainable = False,
            repeat = REPEAT,
            epochs = EPOCHS,
            bs = BS,
            model= BASE_MODEL,
):

    wandb.login()
    # Default hyper-parameters, potentially overridden in sweep mode
    train_dataset = tfds.load(name=DATASET, as_supervised=True, split="train")
    test_dataset = tfds.load(name=DATASET, as_supervised=True, split="test")
    default_config = {
        "batch_size": bs, "epochs": epochs, "dropout": 0.4, "base_model": model, 
        "init_lr": 0.0005, "decay": 0.96, "num_classes": N_CLASSES, "hardware": hw_name, 
        "train_size": len(train_dataset), "test_size": len(test_dataset),
        "dataset": DATASET, "img_dim": IMG_SIZE, "trainable": trainable,
    }

    for _ in range(repeat):
        train(train_dataset, test_dataset, default_config, project=PROJECT, hw=hw_name, team='m1-benchmark')




@call_parse
def main(hw_name:    Param("Name of the hardware: V100, M1, M1Pro, etc...", str)=HW_NAME,
         repeat:     Param("Number of times to repeat training", int)=REPEAT,
         epochs:     Param("Override epochs", int) = EPOCHS,
         bs:         Param("Override Batch Size", int) = BS):

    for model in MODELS:
        for trainable in [False, True]:
            run_one(hw_name=hw_name, trainable=trainable, repeat=repeat, epochs=epochs, bs=bs, model=model)