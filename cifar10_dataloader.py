from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

class CIFAR10DataLoader(Dataset):
    '''A customized version of CIFAR10 dataset for pre-training and evaluating MAE.
    '''
    def __init__(self, data_root, train=True, pretrain=False, transform=None, download=True) -> None:
        super().__init__()
        self.train = train
        self.pretrain = pretrain
        self.transform = transform
        self.data = CIFAR10(root=data_root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int):
        img, target = self.data[index]
        if self.transform:
            img = self.transform(img)
        # in pre-train stage, we only need images as inputs
        if self.pretrain:
            return img
        else:
            return img, target