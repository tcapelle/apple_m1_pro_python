import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10


PROJECT = 'apple_m1_pro'
ENTITY  = 'tcapelle'
GROUP = 'keras'

# Set the random seeds
os.environ['TF_CUDNN_DETERMINISTIC'] = '1' 
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)


import wandb
from wandb.keras import WandbCallback

wandb.login()

# get data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Subsetting train data and normalizing to [0., 1.]
x_train, x_test = x_train[::5] / 255., x_test / 255.
y_train = y_train[::5]

CLASS_NAMES = ["airplane", "automobile", "bird", "cat",
               "deer", "dog", "frog", "horse", "ship", "truck"]

print('Shape of x_train: ', x_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of x_test: ', x_test.shape)
print('Shape of y_test: ', y_test.shape)


def Model():
    inputs = keras.layers.Input(shape=(32, 32, 3))

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(32, activation='relu')(x)

    outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    return keras.models.Model(inputs=inputs, outputs=outputs)

# Initialize wandb with your project name
run = wandb.init(project=PROJECT,
                 entity=ENTITY,
                 group=GROUP,
                 config={  # and include hyperparameters and metadata
                     "learning_rate": 0.005,
                     "epochs": 5,
                     "batch_size": 128,
                     "loss_function": "sparse_categorical_crossentropy",
                     "architecture": "CNN",
                     "dataset": "CIFAR-10"
                 })
config = wandb.config  # We'll use this to configure our experiment

# Initialize model like you usually do.
tf.keras.backend.clear_session()
model = Model()
model.summary()

# Compile model like you usually do.
# Notice that we use config, so our metadata matches what gets executed
optimizer = tf.keras.optimizers.Adam(config.learning_rate) 
model.compile(optimizer, config.loss_function, metrics=['acc'])

# We train with our beloved model.fit
# Notice WandbCallback is used as a regular callback
# We again use config
_ = model.fit(x_train, y_train,
          epochs=config.epochs, 
          batch_size=config.batch_size,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback()])

loss, accuracy = model.evaluate(x_test, y_test)
print('Test Error Rate: ', round((1 - accuracy) * 100, 2))

# With wandb.log, we can easily pass in metrics as key-value pairs.
wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})

run.finish()