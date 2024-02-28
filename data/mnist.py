from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

class DatasetMNIST(Dataset):
    def __init__(self, train=True, transforms = ToTensor()):
        self.dataset =  MNIST(root='data', train=train, download=True)
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        return self.transform(image), label