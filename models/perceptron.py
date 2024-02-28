import torch
from torch import Tensor
from torch import nn

from dataclasses import dataclass

@dataclass
class Dimensions:
    input : int
    hidden  : int
    output : int

class Perceptron(nn.Module):
    def __init__(self, dimensions : Dimensions):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dimensions.input, dimensions.hidden),
            nn.ReLU(),
            nn.Linear(dimensions.hidden, dimensions.output),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input) -> Tensor:
        return self.layers(input)