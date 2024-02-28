from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer

def train(model : Module, optimizer : Optimizer, criterion : Callable[[Tensor, Tensor], Tensor], data : DataLoader):
    average_loss = 0

    model.train(mode=True)
    for batch, (data, target) in enumerate(data):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"Training loss {loss.item():.3f} on batch {batch}")
        average_loss += loss.item()

    return average_loss / len(data)


def test(model : Module, criterion : Callable[[Tensor, Tensor], Tensor], data : DataLoader,):
    average_loss = 0

    model.train(mode=False)
    with torch.no_grad():
        for batch, (data, target) in enumerate(data):
            output = model(data)
            loss = criterion(output, target)

            print(f"Testing loss {loss.item():.3f} on batch {batch}")
        average_loss += loss.item()

    return average_loss / len(data)