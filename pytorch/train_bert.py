import torch, math, wandb, argparse
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from time import perf_counter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, default_data_collator
from datasets import load_dataset


PROJECT = "pytorch-M1Pro"
ENTITY = "capecape"

config_defaults = SimpleNamespace(
    batch_size=8,
    epochs=1,
    num_experiments=1,
    learning_rate=1e-3,
    model_name="bert-base-cased",
    dataset="yelp_review_full",
    device="mps",
    gpu_name="M1Pro GPU 16 Cores",
    num_workers=0
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
    return parser.parse_args()


def get_dls(model_name="bert-base-cased", dataset_name="yelp_review_full", batch_size=8, num_workers=0, sample_size=1000):

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

def to_device(batch, device="cpu"):
    for k,v in batch.items():
        batch[k] = v.to(device)
    return batch

@dataclass
class MicroTrainer:
    model: torch.nn.Module
    train_dl: DataLoader
    test_dl: DataLoader=None
    device: str="mps"

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=config_defaults.learning_rate)
        self.n_steps_per_epoch = math.ceil(len(self.train_dl.dataset) / self.train_dl.batch_size)
        
    def do_one_batch(self, batch):
        batch = to_device(batch, device=self.device)
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def do_one_epoch(self, dl, epoch):
        for step, batch in enumerate(tqdm(dl, leave=False)):
            ti = perf_counter()
            self.optimizer.zero_grad()
            loss = self.do_one_batch(batch)
            loss.backward()
            self.optimizer.step()
            tf = perf_counter()


            self.example_ct += len(batch["labels"])
            metrics = {"train/train_loss": loss, 
                        "train/epoch": (step + 1 + (self.n_steps_per_epoch * epoch)) / self.n_steps_per_epoch, 
                        "train/example_ct": self.example_ct,
                        "sqe_per_sec":len(batch["labels"])/(tf-ti)}
            if step + 1 < self.n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)

            self.step_ct += 1

    def fit(self, epochs):
        self.example_ct = 0
        self.step_ct = 0
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.do_one_epoch(self.train_dl, epoch)
    
    def inference(self, repeat=5):
        self.model.eval()
        batch = next(iter(self.train_dl))
        N = len(batch["labels"])
        inference_times = []
        for _ in tqdm(range(repeat)):
            with torch.inference_mode():
                ti = perf_counter()
                _ = self.do_one_batch(batch)
                tf = perf_counter()
                inference_times.append(N/(tf-ti))
        wandb.log({"inference_seq_per_sec": sum(inference_times)/repeat})

def train_bert(config):
    train_dl, _ = get_dls(
        model_name=config.model_name,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    model = get_model(config.model_name).to(config.device)

    trainer = MicroTrainer(model, train_dl, device=config.device)
    with wandb.init(project=PROJECT, entity=ENTITY, config=config):
        if not config.inference_only:
            trainer.fit(config.epochs)
        trainer.inference()

if __name__ == "__main__":
    args = parse_args()
    for _ in range(args.num_experiments):
        train_bert(config=args)