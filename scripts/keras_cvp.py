import os
import time
import random
import shutil
import tempfile

import wandb
from wandb.keras import WandbCallback
from fastcore.script import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras.backend import count_params
import tensorflow_datasets as tfds

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)


PROJECT = "m1-benchmark"
HW = 'M1Pro'

IMG_DIM = 128
N_CLASSES = 10
DATASET = "cifar10"
BASE_MODEL = "MobileNetV2"


class SamplesSec(K.callbacks.Callback):
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
        self.batch_train_start = time.time()

    def on_train_batch_end(self, batch, logs={}):
        t = time.time() - self.batch_train_start
        self.batch_times.append(t)

    def on_epoch_end(self, epoch, logs={}):
        print(f'\nepoch: {epoch}\n')
        self.batch_times.sort()
        avg_time_per_batch = sum(self.batch_times[0:-self.drop])/(len(self.batch_times)-self.drop)
        samples_s_batch = self.batch_size / avg_time_per_batch
        wandb.log({"samples_per_batch", samples_s_batch}, step=epoch)
        self.samples_s += samples_s_batch
    
    def on_train_end(self, logs={}):
        wandb.log({"samples_per_s": self.samples_s/self.epochs})

def preprocess(image, label=None):
    """Normalize and resize images, one-hot labels""" 
    if label is None:
        label = image['label']
        image = image['image']
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (IMG_DIM, IMG_DIM), method='nearest')
    label = tf.one_hot(label, N_CLASSES)
    return image, label

def prepare(dataset, batch_size=None, cache=True, x=4):
    """Preprocess, shuffle, batch (opt), cache (opt) and prefetch a tf.Dataset"""
    ds = dataset.map(preprocess, num_parallel_calls=x)
    if cache:
        ds = ds.cache(DS_CACHE)
    ds = ds.shuffle(1024)
    if batch_size:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def trainable_params(model):
    """Count the number of trainable parameters in a Keras model"""
    trainable_count = np.sum([count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([count_params(w) for w in model.non_trainable_weights])

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    return trainable_count

def train(train_dataset, test_dataset, default_config, project=PROJECT, hw=HW):
    """Run transfer learning on the configured model and dataset"""
    global IMG_DIM, N_CLASSES, DS_CACHE
    with wandb.init(project=project, group=hw, config=default_config) as run:
        # Set global defaults when running in sweep mode
        IMG_DIM = run.config.img_dim
        N_CLASSES = run.config.num_classes
        DS_CACHE = os.path.join(tempfile.mkdtemp(), str(hash(frozenset(run.config.items()))))

        # Setup base model to transfer from, optionally fine-tune
        base_model = getattr(tf.keras.applications, run.config.base_model)(
            input_shape=(run.config.img_dim, run.config.img_dim, 3),
            include_top=False, weights='imagenet')
        base_model.trainable = run.config.trainable

        # Decay learning rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        run.config.init_lr, decay_steps=run.config.train_size, decay_rate=run.config.decay)

        # Compile model for this dataset
        model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(run.config.dropout),
        tf.keras.layers.Dense(run.config.num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'top_k_categorical_accuracy'])
        
        # Update config and print summary
        run.config.update({
        "total_params": model.count_params(),
        "trainable_params": trainable_params(model),
        })          
        print("Model {}:".format(run.config.base_model))
        print("  trainable parameters:", run.config.trainable_params)
        print("      total parameters:", run.config.total_params)
        print("Dataset {}:".format(run.config.dataset))
        print("  training: ", run.config.train_size)
        print("      test: ", run.config.test_size)
        print("     shape: {}\n".format((run.config.img_dim, run.config.img_dim, 3)))
        
        # Train the model
        train_batches = prepare(train_dataset, batch_size=run.config.batch_size)
        test_batches = prepare(test_dataset, batch_size=run.config.batch_size)

        cbs = [
            wandb.keras.WandbCallback(save_model=False),
            SamplesSec(run.config.epochs, run.config.batch_size)]

        _ = model.fit(train_batches, epochs=run.config.epochs, validation_data=test_batches,
                            callbacks=cbs)
        shutil.rmtree(os.path.dirname(DS_CACHE))


@call_parse
def main(
    project:  Param("Name of the wandb Project to log on", str)='m1-benchmark',
    hw:       Param("Name of the hardware: V100, M1, M1Pro, etc...", str)='M1Pro',
    trainable: Param("Train full model or only head", store_true)=False,
    repeat:    Param("Number of times to repeat training", int)=1,
    epochs:     Param("Override epochs", int) = 10,
    bs: Param("Override Batch Size", int) = 128,
):

    wandb.login()
    # Default hyper-parameters, potentially overridden in sweep mode
    train_dataset = tfds.load(name=DATASET, as_supervised=True, split="train")
    test_dataset = tfds.load(name=DATASET, as_supervised=True, split="test")
    default_config = {
        "batch_size": bs, "epochs": epochs, "dropout": 0.4, "base_model": BASE_MODEL, 
        "init_lr": 0.0005, "decay": 0.96, "num_classes": N_CLASSES, "hardware": hw, 
        "train_size": len(train_dataset), "test_size": len(test_dataset),
        "dataset": DATASET, "img_dim": IMG_DIM, "trainable": trainable,
    }
    
    for _ in range(repeat):
            train(train_dataset, test_dataset, default_config, project=project, hw=hw)