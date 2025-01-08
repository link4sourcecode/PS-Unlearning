import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils import data, models, seed
import argparse

# Settings.
TYPE = ['MNIST', 'CIFAR10', 'CIFAR100', 'TINYIMAGENET']
EPOCH = {'MNIST': 50, 'CIFAR10': 150,
         'CIFAR100': 150, 'TINYIMAGENET': 150}
LR = {'MNIST': 0.01, 'CIFAR10': 0.05,
      'CIFAR100': 0.15, 'TINYIMAGENET': 0.3}
LEN = {'MNIST': 6000, 'CIFAR10': 5000, 'CIFAR100': 500, 'TINYIMAGENET': 500}


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


# Make sure the loaded remain set work properly.
class CustomDataset(Dataset):
    """This is the customized dataset that use to create subset.

    """

    def __init__(self, _data, labels, transform=None):
        self.data = _data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            self.data[idx] = self.transform(self.data[idx])
        return self.data[idx], self.labels[idx]


def read_last_layer_weight(model):
    """This function will return the last layer's weight of the model.
    The shape of the weight is different for different models:
    LeNet5: (10, 64)
    CIFAR10: (10, 64)
    CIFAR100: (100, 64)
    VGG16: (200, 1024)

    Args:
        model (network): The model that we want to get the last layer's weight.
    """

    last_linear_layer_weight = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear_layer_weight = module.weight

    if last_linear_layer_weight is None:
        raise ValueError("No linear layer weight found in the model.")

    return last_linear_layer_weight.detach().numpy()


def load_init_model(model_type):
    """ This function will load the initialized models.

    Args:
        model_type (str): The type of the model.

    Returns:
        model: The loaded model.
    """
    if model_type == "MNIST":
        model_name = "init_mnist.pt"
        model = models.LeNet5()

    elif model_type == "CIFAR10":
        model_name = "init_cifar10.pt"
        model = models.resnet20(3, 10)

    elif model_type == "CIFAR100":
        model_name = "init_cifar100.pt"
        model = models.resnet50(3, 100)

    elif model_type == "TINYIMAGENET":
        model_name = "init_tinyimagenet.pt"
        model = models.VGG16(200)
    else:
        raise ValueError("The model name is not correct.")

    models.load_model(model, model_name)
    return model


def load_original_model(model_type, expid):
    """ This function will load the original models.

    Args:
        model_type (str): The type of the model.
        expid (int): The experiment id. From 0 to 9.

    Returns:
        model: The loaded model.
    """
    assert 0 <= expid <= 9, "The experiment id should be from 0 to 9."

    if model_type == "MNIST":
        model_name = f"original_mnist_{expid}.pt"
        model = models.LeNet5()

    elif model_type == "CIFAR10":
        model_name = f"original_cifar10_{expid}.pt"
        model = models.resnet20(3, 10)

    elif model_type == "CIFAR100":
        model_name = f"original_cifar100_{expid}.pt"
        model = models.resnet50(3, 100)

    elif model_type == "TINYIMAGENET":
        model_name = f"original_tinyimagenet_{expid}.pt"
        model = models.VGG16(200)
    else:
        raise ValueError("The model name is not correct.")

    models.load_model(model, model_name)
    return model


def get_target_features(model, target_class, full_set):
    """This function will return the target class weight and features.

    Args:
        model (network): The model that we want to get the last layer's weight.
        target_class (int): The target class that we want to get the weight and features.
        full_set (dataset): The dataset that we want to get the feature.

    Returns:
        weight: The last layer's weight.
        feature: The feature of the data.
    """
    target_set = data.subset_with_one_label(full_set, target_class)
    target_loader = DataLoader(target_set, batch_size=256, shuffle=False)

    target_features = np.array([models.get_feature(model, target_loader)])
    # Remove the first dimension. e.g. (1, 500, 1024) -> (500, 1024)
    # The shape of the reshaped_features is (data_num, feature_num)
    reshaped_features = np.squeeze(target_features, axis=0)
    return reshaped_features


def NaivePS(model_type, expid, ratio=0.1):
    """This function will return the Naive PS model.

    Args:
        model_type (str): The type of the model.
        expid (int): The experiment id. From 0 to 9.
        ratio (float): The ratio of the unlearn data in the original dataset.

    Returns:
        model: The updated model.
    """
    original_model = load_original_model(model_type, expid)
    init_model = load_init_model(model_type)
    full_training_set, _ = data.load_full_set(model_type)

    # The shape of the weight is (class_num, feature_num)
    # For TinyImageNet, the class_num is 200, feature_num is 1024.
    # For CIFAR100, the class_num is 100, feature_num is 64.
    init_weights = read_last_layer_weight(init_model)
    original_weights = read_last_layer_weight(original_model)
    original_features = None

    filepath = f"./features_{model_type}_{expid}.npy"
    if os.path.exists(filepath):
        print(f"Loading features from {filepath}...")
        original_features = np.load(filepath)

    else:
        print(f"Calcualte the features if not pre-stored.")
        for class_id in range(10):
            print(f"Start processing class {class_id}")
            if original_features is None:
                original_features = get_target_features(
                    original_model, class_id, full_training_set)
            else:
                features = get_target_features(
                    original_model, class_id, full_training_set)
                original_features = np.vstack(
                    (original_features, features))

        np.save(filepath, original_features)
        print(f"Features saved to {filepath}.")

    print(f"Start NaivePS for {model_type}...")
    original_pinv = np.linalg.pinv(original_features)
    original_theta = np.dot((original_weights - init_weights), original_pinv)

    # Unlearn samples come from class 1, thus we can calculate the position directly.
    # Namely remove the last 10% of the target class data. (from both weight and feature)
    # The ratio is forget ratio, while the number is for remain set.
    num_remain = int(LEN[model_type] * (1 - ratio))
    index_start = LEN[model_type] + num_remain - 1
    index_end = 2 * LEN[model_type] - 1

    new_theta = np.delete(original_theta, np.s_[index_start:index_end], axis=1)
    new_feature = np.delete(original_features, np.s_[
                            index_start:index_end], axis=0)

    new_weight = init_weights + np.dot(new_theta, new_feature)

    updated_model = models.replace_last_layer_weight(
        original_model, new_weight)
    models.save_model(updated_model, f"updated_{model_type}_{expid}.pt")
    print(f"NaivePS for {model_type} is done.")


def relabel_set(data_type, expid, ratio=0.1):
    """This function will relabel the unlearn data in the dataset.

    Args:
        data_type (str): The dataset that we want to relabel.
        expid (str): The experiment id that we want to relabel.
        ratio (float, optional): The ratio of the unlearn data in the original dataset. Defaults to 0.1.
    """
    # Load the unlearn set.
    unlearn_set = data.load_unlearn_set(data_type, ratio)
    unlearn_loader = DataLoader(unlearn_set, batch_size=256, shuffle=False)

    # Load original model and get original logits.
    original_model = load_original_model(data_type, expid)
    _, original_logits = models.get_feature_and_logits(
        original_model, unlearn_loader)

    new_logtis = np.copy(original_logits)
    original_logits[:, 1] = 0
    delta_y = new_logtis[:, 1]
    new_logtis[:, 1] = -9999

    # Allocate the delta y to the second and third largest class.
    # Since the target class is set to 0, we need to find the largest and second large class.
    sorted_indices = np.argsort(original_logits, axis=1)
    second_largest_indices = sorted_indices[:, -1]
    third_largest_indices = sorted_indices[:, -2]
    second_largest_values = original_logits[np.arange(
        len(original_logits)), second_largest_indices]
    third_largest_values = original_logits[np.arange(
        len(original_logits)), third_largest_indices]
    proportions = softmax(
        np.vstack((second_largest_values, third_largest_values)).T)
    Delta_y_distributed = delta_y[:, None] * proportions
    new_logtis[np.arange(len(new_logtis)),
               second_largest_indices] += Delta_y_distributed[:, 0]
    new_logtis[np.arange(len(new_logtis)),
               third_largest_indices] += Delta_y_distributed[:, 1]
    new_label = softmax(new_logtis)

    relabeled_set = data.RelabeledSet(unlearn_set, new_label)
    return relabeled_set


def PS(data_type, expid, ratio=0.1):
    """This function will relabel the unlearn data in the dataset and fine-tune the model.

    Args:
        data_type (str): The dataset that we want to relabel.
        expid (str): The experiment id that we want to relabel.
        ratio (float, optional): The ratio of the unlearn data in the original dataset. Defaults to 0.1.
    """
    # Relabel the unlearn set.
    print(f"Start relabel unlearning data for {data_type}...")
    relabeled_set = relabel_set(data_type, expid, ratio)
    relabeled_loader = DataLoader(relabeled_set, batch_size=256, shuffle=True)
    print(f"Relabel unlearning data for {data_type} is done.")

    # Load unlearn set to check the accuracy curve.
    unlearn_set = data.load_unlearn_set(data_type)
    unlearn_loader = DataLoader(unlearn_set, batch_size=256, shuffle=False)

    # Load remain set to check the accuracy curve.
    remain_set = data.load_remain_set(data_type)
    remain_loader = DataLoader(remain_set, batch_size=256, shuffle=False)

    # Load the original model. 
    updated_model = load_original_model(data_type, expid)
    is_train = False  

    _, full_test_set = data.load_full_set(data_type)
    full_test_loader = DataLoader(full_test_set, batch_size=256, shuffle=False)
    save_name = f"PS_{data_type}_{expid}.pt"
    print(f"Start PS {save_name}")
    models.fine_tune_relabel(updated_model, relabeled_loader, unlearn_loader, remain_loader, full_test_loader,
                             EPOCH[data_type], LR[data_type], save_name, is_train)
    print(f"PS for {data_type} is done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NaivePS and Relabeling")
    parser.add_argument("--model_type", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100", "TINYIMAGENET"],
                        help="Type of the dataset/model")
    parser.add_argument("--expid", type=int, default=0,
                        help="Experiment ID (from 0 to 9)")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Ratio of the unlearn data in the original dataset")
    parser.add_argument("--seed", type=int, default=209,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    seed.seed_everything(args.seed)
    NaivePS(args.model_type, args.expid, args.ratio)
    PS(args.model_type, args.expid, args.ratio)