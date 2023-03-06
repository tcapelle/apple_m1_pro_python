## Author: Thomas Capelle, Soumik Rakshit
## Mail:   tcapelle@wandb.com, soumik.rakshit@wandb.com

""""Benchmarking apple M1Pro with Tensorflow
@wandbcode{apple_m1_pro}"""

import wandb, argparse
from types import SimpleNamespace
from time import perf_counter

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import legacy as legacy_optimizers
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DefaultDataCollator
from datasets import load_dataset

import wandb
from wandb.keras import WandbCallback

from utils import get_apple_gpu_name


PROJECT = "pytorch-M1Pro"
ENTITY = "capecape"
GROUP = "tf"

config_defaults = SimpleNamespace(
    batch_size=4,
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    model_name="bert-base-cased",
    dataset="yelp_review_full",
    device="mps",
    gpu_name=get_apple_gpu_name(),
    num_workers=0,
    mixed_precision=False,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config_defaults.batch_size)
    parser.add_argument('--epochs', type=int, default=config_defaults.epochs)
    parser.add_argument('--num_experiments', type=int, default=config_defaults.num_experiments)
    parser.add_argument('--learning_rate', type=float, default=config_defaults.learning_rate)
    parser.add_argument('--model_name', type=str, default=config_defaults.model_name)
    parser.add_argument('--dataset', type=str, default=config_defaults.dataset)
    parser.add_argument('--device', type=str, default=config_defaults.device)
    parser.add_argument('--gpu_name', type=str, default=config_defaults.gpu_name)
    parser.add_argument('--num_workers', type=int, default=config_defaults.num_workers)
    parser.add_argument('--inference_only', action="store_true")
    parser.add_argument('--mixed_precision', action="store_true")
    return parser.parse_args()

class SamplesSec(tf.keras.callbacks.Callback):
    def __init__(self, batch_size=1, drop=5):
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
        wandb.log({"samples_per_sec": self.batch_size/t})


def get_dls(model_name="bert-base-cased", dataset_name="yelp_review_full", batch_size=8, num_workers=0, sample_size=100):

    # download and prepare cc_news dataset
    dataset = load_dataset(dataset_name)

    # get bert and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    # tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(sample_size))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(sample_size))

    default_data_collator = DefaultDataCollator(return_tensors="tf")

    train = small_train_dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    validation = small_eval_dataset.to_tf_dataset(
        columns=["input_ids", "token_type_ids", "attention_mask"],
        label_cols=["labels"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return train, validation


def get_model(model_name="bert-base-cased", num_labels=5):
    return TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def train_bert(config):
    train_ds, _ = get_dls(
        model_name=config.model_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    if config.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')

    optimizer = legacy_optimizers.Adam(learning_rate=config.learning_rate)

    model = get_model(config.model_name)

    model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["accuracy"],
    )

    with wandb.init(project=PROJECT, entity=ENTITY, group=GROUP, config=config):
        model.fit(
            train_ds,
            epochs=config.epochs,
            callbacks=[WandbCallback(save_model=False),
                       SamplesSec(config.batch_size)],
        )

if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train_bert(config=args)