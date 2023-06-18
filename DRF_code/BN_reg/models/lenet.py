import math
import torch
import torch.nn as nn

__all__ = ['LeNet']

defaultcfg = {18: [6, 16]}


class LeNet(nn.Module):
    def __init__(self, dataset='data.cifar10', depth=18, cfg=None):
        # super函数解决在多种继承中更好的调用父类super在多继承中经常使用
        super(LeNet, self).__init__()

        if cfg is None:
            cfg = defaultcfg[depth]

        self.conv1 = nn.Conv2d(1, cfg[0], kernel_size=(5, 5), stride=(1, 1), padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        # ==========

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(5 * 5 * cfg[1], 120)
        self.linear2 = nn.Linear(120, 84)
        # if dataset == 'data.cifar10':
        #     self.num_classes = 10
        # elif dataset == 'cifar100':
        #     self.num_classes = 100
        self.linear3 = nn.Linear(84, 10)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        x = self.maxpool2(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

