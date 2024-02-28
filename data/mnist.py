from typing import Tuple

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

class MNISTDataset(Dataset):
    train = MNIST(root='data', train=True, download=True)
    test = MNIST(root='data', train=False, download=True)

    def __init__(self, train=True, transforms = ToTensor()):
        self.dataset = self.train if train else self.test
        self.transform = transforms

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        return self.transform(image), label