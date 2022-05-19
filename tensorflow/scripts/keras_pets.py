import os
import random
import wandb

import wandb
from wandb.keras import WandbCallback

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)


PROJECT = 'apple_m1_pro'
GROUP = 'M1Pro'

IMG_SIZE = 128
BS = 32
EPOCHS = 10
LR = 1e-3

def train():
    wandb.login()
    run = wandb.init(project=PROJECT, group=GROUP)
    tf.keras.backend.clear_session()
    model = make_model(input_shape=(IMG_SIZE, IMG_SIZE) + (3,), num_classes=2)

    train_ds, val_ds = prepare_data()

    callbacks = [
        WandbCallback(save_model=False)
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=EPOCHS, callbacks=callbacks, validation_data=val_ds,
    )

    run.log({'batch_size':BS, 'epochs':EPOCHS, 'image_size':IMG_SIZE, 'lr': LR})
    run.finish()

def prepare_data():
    artifact = wandb.use_artifact('tcapelle/apple_m1_pro/PetImages:latest', type='dataset')
    dataset_dir = artifact.download()

    def get_data(data_dir):
        print(f'Reading from : {Path(data_dir).absolute()}')
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            Path(data_dir),
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=(IMG_SIZE,IMG_SIZE),
            batch_size=BS,
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            Path(data_dir),
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=(IMG_SIZE,IMG_SIZE),
            batch_size=BS,
        )
        return train_ds, val_ds

    train_ds, val_ds = get_data(dataset_dir)

    #speed up dataloading
    # train_ds = train_ds.prefetch(buffer_size=BS)
    # val_ds = val_ds.prefetch(buffer_size=BS)
    
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds

def make_model(input_shape, num_classes):
    
    #no data augment for now
    data_augmentation = lambda x: x

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


if __name__=='__main__':
    train()