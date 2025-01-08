import pickle
import os

from torchvision.datasets import MNIST, CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, Subset


TRANS_MNIST = transforms.Compose([transforms.Resize((32, 32)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307, ), (0.3081, ))])
TRANS_CIFAR10_TRAIN = transforms.Compose([transforms.ToTensor(),
                                          transforms.RandomCrop(32, 4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Normalize((0.485, 0.456, 0.406), \
                                                               (0.229, 0.224, 0.225))])
TRANS_CIFAR10_TEST = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), \
                                                              (0.229, 0.224, 0.225))])

TRANS_CIFART100_TRAIN = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomCrop(32, 4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize((0.507, 0.487, 0.441),
                                                                 (0.267, 0.256, 0.276))])

TRANS_CIFAR100_TEST = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.507, 0.487, 0.441),
                                                               (0.267, 0.256, 0.276))])

TRANS_TINYIMAGENET_TRAIN = transforms.Compose([transforms.ToTensor(),
                                               transforms.RandomCrop(64, 4),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.Normalize((0.480, 0.448, 0.398),
                                                                    (0.277, 0.269, 0.282))])

TRANS_TINYIMAGENET_TEST = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.480, 0.448, 0.398),
                                                                   (0.277, 0.269, 0.282))])

# DATA_PATH = 'path\\to\\your\\datasets'
DATA_PATH = "D:\\Doin4\\Code\\Fraud Unlearning EXP\\PS-Unlearning-Master\\datasets"


class CustomDataset(Dataset):
    """This is the customized dataset that use to create subset.

    """

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            self.data[idx] = self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]


class RelabeledSet(Dataset):
    def __init__(self, old_set, new_labels):
        self.old_set = old_set
        self.new_labels = new_labels
        self.transform = old_set.transform

    def __getitem__(self, idx):
        data, _ = self.old_set[idx]
        return data, self.new_labels[idx]

    def __len__(self):
        return len(self.old_set)


def load_full_mnist(data_path):
    """This function use torchvision to load the full MNIST data. With train and test set.

    Returns:
        train_set (data): The training part of MNIST.
        test_set (data): The testing part of MNIST.
    """
    train_set = MNIST(root=data_path, train=True, download=False,
                      transform=TRANS_MNIST)
    test_set = MNIST(root=data_path, train=False, download=False,
                     transform=TRANS_MNIST)
    return train_set, test_set


def load_full_cifar10(data_path):
    """This function use torchvision to load the full Cifar-10 data. With train and test set.

    Returns:
        train_set (data): The training part of Cifar-10.
        test_set (data): The testing part of Cifar-10.
    """
    train_set = CIFAR10(root=data_path, train=True, download=False,
                        transform=TRANS_CIFAR10_TRAIN)
    test_set = CIFAR10(root=data_path, train=False, download=False,
                       transform=TRANS_CIFAR10_TEST)
    return train_set, test_set


def load_full_cifar100(data_path):
    """This function use torchvision to load the full CIFAR-100 data. With train and test set.

    Args:
        data_path (_type_): The path to the dataset.

    Returns:
        train_set (data): The training part of CIFAR-100.
        test_set (data): The testing part of CIFAR-100.
    """
    train_set = CIFAR100(root=data_path, train=True, download=False,
                         transform=TRANS_CIFART100_TRAIN)
    test_set = CIFAR100(root=data_path, train=False, download=False,
                        transform=TRANS_CIFAR100_TEST)

    return train_set, test_set


def load_full_tinyimagenet(data_path):
    """This function use ImageFolder to load the full TinyImageNet data. With train and test set.

    Args:
        data_path (_type_): The path to the dataset.

    Returns:
        train_set (data): The training part of TinyImageNet.
        test_set (data): The testing part of TinyImageNet.
    """
    train_path = os.path.join(data_path, 'tiny-imagenet-200', 'train')
    # Note the TinyImageNet don't have labeled test set, thus, we can only use the validation set as test set.
    test_path = os.path.join(data_path, 'tiny-imagenet-200', 'val')
    train_set = ImageFolder(
        root=train_path, transform=TRANS_TINYIMAGENET_TRAIN)
    test_set = ImageFolder(root=test_path, transform=TRANS_TINYIMAGENET_TEST)

    return train_set, test_set


def subset_with_one_label(full_set, label):
    """This function can construct a dataset with only one label.

    Args:
        full_set (dataset): The full dataset.
        label (int): The label that we want.
        is_mnist (bool, optional): Choose dataset. Defaults to True.

    Returns:
        subset (dataset): The dataset with only one label.
    """

    subset = Subset(full_set,
                    indices=[i for i in range(len(full_set)) if
                             full_set[i][1] == label])

    filtered_data = []
    filtered_label = []
    for data_point, label_point in subset:
        filtered_data.append(data_point)
        filtered_label.append(label_point)

    new_subset = CustomDataset(filtered_data, filtered_label)

    return new_subset



def load_unlearn_set(data_type: str, ratio: float = 0.1):
    """ This function will load the unlearned data for the given data type and ratio.

    Args:
        data_type (str): The type of the data.
        ratio (float): The ratio of the unlearned data.
    """
    if data_type == "MNIST":
        unlearn_path = f"{DATA_PATH}\\MNIST_unlearn_{ratio}.pkl"
    elif data_type == "CIFAR10":
        unlearn_path = f"{DATA_PATH}\\CIFAR10_unlearn_{ratio}.pkl"
    elif data_type == "CIFAR100":
        unlearn_path = f"{DATA_PATH}\\CIFAR100_unlearn_{ratio}.pkl"
    elif data_type == "TINYIMAGENET":
        unlearn_path = f"{DATA_PATH}\\TINYIMAGENET_unlearn_{ratio}.pkl"
    else:
        raise ValueError("The data type is not correct.")

    with open(unlearn_path, 'rb') as file:
        unlearn_set = pickle.load(file)

    return unlearn_set


def load_remain_set(data_type: str, ratio: float = 0.1):
    """ This function will load the remained data for the given data type and ratio.

    Args:
        data_type (str): The type of the data.
        ratio (float): The ratio of the remained data.
    """
    if data_type == "MNIST":
        remain_path = f"{DATA_PATH}\\MNIST_remain_{ratio}.pkl"
    elif data_type == "CIFAR10":
        remain_path = f"{DATA_PATH}\\CIFAR10_remain_{ratio}.pkl"
    elif data_type == "CIFAR100":
        remain_path = f"{DATA_PATH}\\CIFAR100_remain_{ratio}.pkl"
    elif data_type == "TINYIMAGENET":
        remain_path = f"{DATA_PATH}\\TINYIMAGENET_remain_{ratio}.pkl"
    else:
        raise ValueError("The data type is not correct.")

    with open(remain_path, 'rb') as file:
        remain_set = pickle.load(file)

    return remain_set


def load_full_set(data_type: str):
    """This function use data_type to load the full dataset.

    Args:
        data_type (str): The target dataset.
    """
    if data_type == "MNIST":
        return load_full_mnist(DATA_PATH)
    elif data_type == "CIFAR10":
        return load_full_cifar10(DATA_PATH)
    elif data_type == "CIFAR100":
        return load_full_cifar100(DATA_PATH)
    elif data_type == "TINYIMAGENET":
        return load_full_tinyimagenet(DATA_PATH)
    else:
        raise ValueError("The data type is not correct.")


if __name__ == "__main__":
    pass
