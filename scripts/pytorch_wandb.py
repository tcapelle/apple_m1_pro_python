import os
import random

import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from fastprogress import progress_bar, master_bar

PROJECT = "apple_m1_pro"
ENTITY = "tcapelle"
GROUP = 'pytorch'

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.login()

run = wandb.init(project=PROJECT,
                 entity=ENTITY,
                 group=GROUP, 
                 config = {
                    "lr": 0.005,
                    "epochs": 5,
                    "batch_size": 128,
                    "loss_function": "CrossEntropyLoss",
                    "architecture": "cnn",
                    "dataset": "CIFAR-10",
                })
config = wandb.config

transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def get_data(train=True, sample=True):
    ds = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=True, transform=transform)

    if sample and train:
        ds.data = ds.data[::5]

    loader = torch.utils.data.DataLoader(
        ds, batch_size=config.batch_size*(2-train), shuffle=train, num_workers=2)
    return loader

train_loader, test_loader = get_data(True), get_data(False)

train_loader = get_data(True)
test_loader = get_data(False)

x,y = next(iter(train_loader))
print(f'Input shape: {x.shape}')

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 32, 3),
    nn.ReLU(),
    nn.Conv2d(32, 32, 3),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 32),
    nn.ReLU(),
    nn.Linear(32, len(classes))
).to(device)

class Learner:
    "A Wrapper around model and data"
    def __init__(self, train_loader, test_loader, model, criterion):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model.to(device)
        self.criterion = criterion
        self.mb = None
        self.batch_ct = 0
        self.example_ct = 0
        
        
    def one_batch_train(self, images, labels):
        "Do one batch train"
        images, labels = images.to(device), labels.to(device)

        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # Forward pass ➡
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward pass ⬅
        loss.backward()

        # Step with optimizer
        self.optimizer.step()

        return loss
    
    def one_epoch_train(self):
        "Do one epoch train"
        self.model.train()
        for images, labels in progress_bar(self.train_loader, parent=self.mb):
            loss = self.one_batch_train(images, labels)
            self.batch_ct += 1
            self.example_ct += len(labels)
    
            # Report metrics every 25th batch
            if ((self.batch_ct + 1) % 25) == 0:
                wandb.log({"epoch": self.epoch, "loss": float(loss)})
                
            self.mb.child.comment = f'train_loss={loss.item():.3f}'
    
    
    def one_batch_test(self, images, labels):
        "Do one batch test"
        images, labels = images.to(device), labels.to(device)

        # Forward pass ➡
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        
        correct = (predicted == labels).sum().item()
        
        return loss, correct
    
    def one_epoch_test(self):
        self.model.eval()
        
        # Run the model on some test examples
        with torch.no_grad():
            correct_total, loss_test = 0, 0
            for images, labels in progress_bar(test_loader, parent=self.mb):
                loss, correct = self.one_batch_test(images, labels)
                correct_total += correct
                wandb.log({"test_loss": float(loss)})
        
        wandb.log({"test_accuracy": correct_total / len(test_loader)})

    
    def save(self):
        # save and log last mdoel to wandb
        torch.save(self.model.state_dict(), 'model.pt')
        wandb.save('model.pt')
    
    def fit(self, epochs, lr=config.lr):
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.mb = master_bar(range(epochs))
        
        wandb.watch(self.model, self.criterion, log='all', log_freq=10)
                          
        for self.epoch in self.mb:
            self.one_epoch_train()
            self.one_epoch_test()
        
        self.save()
            

criterion = nn.CrossEntropyLoss()

learn = Learner(train_loader, test_loader, model, criterion)

learn.fit(config.epochs)

wandb.finish()