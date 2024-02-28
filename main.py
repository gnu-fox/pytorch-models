from typing import Callable

from models.perceptron import Perceptron, Dimensions
from data.mnist import MNISTDataset

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.nn import NLLLoss
from torch.nn import Sequential

def train(model : Module, optimizer : Optimizer, data : DataLoader, criterion : Callable[[Tensor, Tensor], Tensor]):
    for batch, (data, target) in enumerate(data):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f"Training loss {loss.item():.3f} on batch {batch}")

def test(model : Module, data : DataLoader, criterion : Callable[[Tensor, Tensor], Tensor]):
    model.train(mode=False)
    with torch.no_grad():
        for batch, (data, target) in enumerate(data):
            output = model(data)
            loss = criterion(output, target)
            print(f"Testing loss {loss.item():.3f} on batch {batch}")

def main():
    train_dataloader = DataLoader(dataset = MNISTDataset(), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset = MNISTDataset(train=False), batch_size=64, shuffle=True)

    
    model = Perceptron(Dimensions(input=28*28, hidden=128, output=10))
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = NLLLoss()

    model.train(mode=True)
    for epoch in range(5):
        print(f"Epoch {epoch}")
        train(model, optimizer, train_dataloader, criterion)

main()