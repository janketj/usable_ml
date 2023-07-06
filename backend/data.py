from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False, indexed=False):
        super(MNIST, self).__init__(
            root, train=train, transform=transform, download=download
        )
        self.indexed = indexed

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        ret = (index, (img, target)) if self.indexed else (img, target)
        return ret


def get_dataset(test: bool = False, indexed: bool = False) -> Dataset:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    return MNIST(
        root="",
        train=not test,
        download=True,
        transform=transform,
        indexed=indexed,
    )


def get_data_loaders(batch_size: int, test: bool = False):
    dataset = get_dataset(test=test)
    shuffle = not test
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
