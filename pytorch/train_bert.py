## Author: Thomas Capelle, Soumik Rakshit
## Mail:   tcapelle@wandb.com, soumik.rakshit@wandb.com

""""Benchmarking apple M1Pro with Tensorflow
@wandbcode{apple_m1_pro}"""


import torch, wandb, argparse
from types import SimpleNamespace
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from datasets import load_dataset

from utils import MicroTrainer, get_gpu_name


PROJECT = "pt2"
ENTITY = "capecape"
GROUP = "transformers"


config_defaults = SimpleNamespace(
    batch_size=16,
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    model_name="bert-base-cased",
    dataset="yelp_review_full",
    device="mps",
    gpu_name=get_gpu_name(),
    num_workers=8,
    mixed_precision=False,
    syncro=False,
    inference_only=False,
    compile=False,
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
    parser.add_argument('--compile', action="store_true")
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--syncro', action="store_true")
    return parser.parse_args()


def get_dls(model_name="bert-base-cased", dataset_name="yelp_review_full", batch_size=8, num_workers=0, sample_size=2048):

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

    train_loader = DataLoader(
        small_train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_data_collator,
    )

    test_loader = DataLoader(
        small_eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=default_data_collator,
    )

    return train_loader, test_loader


def get_model(model_name="bert-base-cased", num_labels=5):
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def check_cuda(config):
    if torch.cuda.is_available():
        config.device = "cuda"
        config.mixed_precision = True
        config.pt_version = torch.__version__
        config.cuda_version = torch.version.cuda
    return config

def train_bert(config):
    train_dl, test_loader = get_dls(
        model_name=config.model_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    model = get_model(config.model_name).to(config.device)
    if torch.__version__ >= "2.0" and config.compile:
        print("Compiling model...")
        model = torch.compile(model)

    trainer = MicroTrainer(model, train_dl, device=config.device, mixed_precision=config.mixed_precision, syncro=config.syncro)
    tags = [f"pt{torch.__version__}", f"cuda{torch.version.cuda}"] + (config.tags.split(",") if config.tags is not None else [])
    with wandb.init(project=PROJECT, entity=ENTITY, group=GROUP, tags=tags, config=config):
        config = wandb.config
        if not config.inference_only:
            trainer.fit(config.epochs)
        trainer.inference(test_loader)

if __name__ == "__main__":
    args = parse_args()
    args = check_cuda(args)
    for _ in range(args.num_experiments):
        train_bert(config=args)
