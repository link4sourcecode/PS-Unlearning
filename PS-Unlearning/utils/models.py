'''
Author       : Doin4
Date         : 2023-09-02 16:48:49
LastEditors: Doin4
LastEditTime: 2024-10-22 18:32:33
Description  : This file stored models that will be used in this project.
               With basic train and test function.
'''

import time
import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

SAVE_PATH = "D:\\Doin4\\Code\\Fraud Unlearning EXP\\save\\"
# SAVE_PATH = "path\\to\\your\\save\\"

def _weights_init(model):
    """Initialize the weight.

    Args:
        model (model): The network that will be initialized.
    """
    if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        init.kaiming_normal_(model.weight)


class LambdaLayer(nn.Module):
    """Construct lambda layer for the easy implementation.

    Args:
        nn (string): The name of the function.
    """

    def __init__(self, _lambda):
        super(LambdaLayer, self).__init__()
        self._lambda = _lambda

    def forward(self, x):
        """Define the forward function for lambda layer.

        Args:
            x (data): The input data.

        Returns:
            output : The result of this layer.
        """
        return self._lambda(x)


class BasicBlock(nn.Module):
    """This is the basic block of resnet.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0,
                                                                      planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        """The forward function.

        Args:
            x (data): Input data.

        Returns:
            out (data): The model output.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """Construct a new ResNet.

    """

    def __init__(self, dim, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(dim, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """The forward function. The last but one layer's out put is also returned.

        Args:
            x (data): The input data.

        Returns:
            out (data): The output of the model.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        # Modified.
        out1 = out.view(out.size(0), -1)
        out = self.linear(out1)
        return out, out1


def resnet20(dim, classes=10):
    """Construct resnet20.

    Returns:
        model : The resnet20 model.
    """
    return ResNet(dim, BasicBlock, [3, 3, 3], classes)


def resnet50(dim, classes=100):
    """Construct resnet56.

    Returns:
        model : The resnet56 model.
    """
    return ResNet(dim, BasicBlock, [9, 9, 9], classes)


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),)
        self.subsampel1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),)
        self.subsampel2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.L1 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120, 64)
        self.relu1 = nn.ReLU()
        self.L3 = nn.Linear(64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.subsampel1(out)
        out = self.layer2(out)
        out = self.subsampel2(out)
        out = out.reshape(out.size(0), -1)
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out1 = self.relu1(out)
        out = self.L3(out1)
        return out, out1


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i])
              for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block(
            [64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(2*2*512, 4096)
        self.layer7 = vgg_fc_layer(4096, 1024)

        # Final layer
        self.layer8 = nn.Linear(1024, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out1 = self.layer7(out)
        out = self.layer8(out1)

        return out, out1


def accuracy(output, truth):
    """Calculate the accuracy of the trained model.

    Args:
        output (output): The output of the model that will be tested.
        truth (labels): The ground truth of the input image.

    Returns:
        acc (float): The accuracy of the model.
    """
    if (len(output) > 1 and output.shape[0] > 1):
        output = output.argmax(axis=1)
        cmp = output == truth
        acc = cmp.sum() / len(truth)
    else:
        acc = output == truth
    return acc


def load_model(net, name):
    """This function can load the pre-trained parameters to the model.

    Args:
        net (model): The model that will be loaded.
        name (string): The name of the parameter dict.
    """
    net.load_state_dict(torch.load(SAVE_PATH + name))


def load_model_by_dict(net, dict):
    """This function use dict to initialize model

    Args:
        net (model): The model that will be loaded.
        dict (state_dict): The parameters.

    Returns:
        net (model): The loaded model.
    """
    net.load_state_dict(dict)
    return net


def save_model(net, name):
    """This function can save the model.

    Args:
        net (model): The model that will be saved.
        name (string): The name of the saved path.
    """
    torch.save(net.state_dict(), SAVE_PATH + name)


def FT_relabel(net, train_iter, un_test_iter, re_test_iter, full_test_iter, num_epoch, learning_rate, save_name, is_train=True, device=torch.device("cuda")):  # pylint: disable=too-many-locals, too-many-arguments
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(
        0.9, 0.999), weight_decay=1e-4, eps=1e-8)
    loss = nn.CrossEntropyLoss()
    print("start FineTuning----------")
    if is_train:
        net.train()
    else:
        net.eval()

    for epoch in range(num_epoch):
        total_loss = 0
        begin = time.time()
        for inputs, labels in train_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, _ = net(inputs)

            outputs = torch.softmax(outputs, dim=1)

            loss_value = loss(outputs, labels)
            total_loss += loss_value.item()

            loss_value.backward()
            optimizer.step()
        end = time.time()
        # It is normal to have a high loss value, the loss on test set is more important.
        print(f'At training stage, epoch: {epoch + 1},  time: {end - begin}s.')
        total_loss = total_loss / len(train_iter)
        torch.save(net.state_dict(), SAVE_PATH + save_name)


def test(net, test_iter, device=torch.device('cuda')):  # pylint: disable=no-member
    """A simple function to test the network.

    Args:
        net (model): The model that will be tested.
        test_iter (data): The wrapped test data.
        device (string, optional): The device that you want to use.
                                   Defaults to torch.device('cuda').

    Returns:
        acc (float): The final accuracy of the model.
    """
    print("start testing ----------")
    acc = 0
    total_loss = 0
    loss = nn.CrossEntropyLoss()
    net.to(device)
    net.eval()
    with torch.no_grad():
        for inputs, labels in test_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = net(inputs)
            loss_value = loss(outputs, labels)
            total_loss += loss_value.item()
            acc += accuracy(outputs, labels)
    acc = acc / len(test_iter)
    total_loss = total_loss / len(test_iter)
    print(f'test accuracy: {acc}, test loss: {total_loss}')
    print("-------------end testing")
    return acc, total_loss


def fine_tune_relabel(net, loader, un_test_loader, re_test_loader, full_test_loader, epoch_num, lr, save_name, is_train, device='cuda'):
    length = len(list(net.parameters()))
    for i, param in enumerate(net.parameters()):
        if i == length - 1:
            param.requires_grad = True
        else:
            param.requires_grad = False
    FT_relabel(net, loader, un_test_loader, re_test_loader, full_test_loader,
               epoch_num, lr, save_name, is_train, device)

def replace_last_layer_weight(model, new_weight):
    """
    This function replaces the last linear layer's weight of the given model
    with the provided `new_weight`. The shape of `new_weight` must match the
    shape of the original weight.

    Args:
        model (nn.Module): The model whose last layer's weight needs to be replaced.
        new_weight (numpy.ndarray): The new weight to assign to the last layer.

    Returns:
        nn.Module: The updated model with the replaced weight.
    """
    last_linear_layer = None

    # Find the last linear layer in the model
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear_layer = module

    if last_linear_layer is None:
        raise ValueError("No linear layer found in the model.")

    # Check if the new weight shape matches the last layer's weight shape
    original_weight_shape = last_linear_layer.weight.shape
    if new_weight.shape != original_weight_shape:
        raise ValueError(
            f"Shape mismatch: new weight shape {new_weight.shape} does not match "
            f"original weight shape {original_weight_shape}."
        )

    # Replace the weight of the last linear layer
    with torch.no_grad():  # Ensure no gradients are computed for this update
        last_linear_layer.weight.copy_(torch.tensor(new_weight, dtype=last_linear_layer.weight.dtype))

    return model

def get_feature_and_logits(model, loader, device_idx=1):
    """This function helps me to get features.

    Args:
        model (model): The target model.
        loader (loader): The target data loader.

    Returns:
        weighted_feature (feature): The weighted feature.
    """
    if device_idx == 1:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device)
    model.eval()
    # Change log.
    # model.train()
    features = []
    logits = []

    for inputs, labels in loader:
        if device_idx == 1:
            inputs = inputs.to(device)
            labels = labels.to(device)
        logit, feature = model(inputs)
        features.append(feature.cpu().detach().tolist())
        logits.append(logit.cpu().detach().tolist())

    features = np.concatenate(features, axis=0)
    logits = np.concatenate(logits, axis=0)

    return features, logits

if __name__ == "__main__":
    pass