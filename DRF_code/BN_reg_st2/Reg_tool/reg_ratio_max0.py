import torch
import numpy as np
import argparse

default_cfg = {
    'v16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'v19': [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512],
    'r18': [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
    'con': [32, 32, 32]}

default_ratio = {
    # 'v16': [0., 0., 0.1, 0.1, 0.53, 0.53, 0.53, 0.65, 0.65, 0.65, 0.53, 0.53, 0.],
    'v16': [0., 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.38, 0.38, 0.38, 0.18, 0.18, 0.],
    # 'v16': [0., 0.605, 0.605, 0.605, 0.605, 0.605, 0.605, 0.407, 0.407, 0.407, 0.21, 0.21, 0.],
    'v19': [0., 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.21, 0.21, 0.21, 0.],
    'r18': [0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    # 'r18': [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],

}

parser = argparse.ArgumentParser()
parser.add_argument('--pattern', type=str, default='v16', help='the pattern for net param')
args = parser.parse_args(args=[])


def get_conv_weight(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list


def get_bn_weight(model):
    bn_weight_list = []

    for name, param in model.named_parameters():
        if 'bn' in name and 'weight' in name:
            weight = (name, param)
            bn_weight_list.append(weight)

    return bn_weight_list


def filter_gl(model):
    conv_weight_list = get_conv_weight(model)
    filter_gl_list = torch.tensor([]).cuda()

    for name, wight in conv_weight_list:
        ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(wight, 2), dim=[1, 2, 3]))
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return filter_gl_list


def bn_ght(model):

    bn_weight_list = get_bn_weight(model)
    ratio = default_ratio[args.pattern]
    cfg = default_cfg[args.pattern]
    list_n = []

    for t in range(len(bn_weight_list)):
        rank_part = bn_weight_list[t]
        rank_part[1].clone().detach().requires_grad_(True)

        y, i = torch.sort(torch.abs(rank_part[1]))
        thre_index = int(ratio[t] * cfg[t])
        thre = y[thre_index]
        list_n.append(thre)

    return list_n


def bn_each_weight(model, thr):

    bn_weight_list = get_bn_weight(model)
    weight_list2 = torch.tensor([]).cuda()
    thrt = bn_ght(model)
    for t in range(len(bn_weight_list)):
        rank_part = bn_weight_list[t]

        max_num = thrt[t]
        weight = torch.mul(torch.sub(1, torch.div(torch.abs(rank_part[1]), max_num)), thr)
        weight_list2 = torch.cat([weight_list2, weight], dim=0)

    xx = weight_list2.detach().cpu()
    reg_new = np.maximum(np.array(xx), 0)
    return reg_new


def bn_change_loss(model, thr):

    num = bn_each_weight(model, thr)
    numm = torch.tensor(num).cuda()
    gl = filter_gl(model)
    loss_tol = sum(torch.mul(numm, gl))

    return loss_tol


def filter_l1(model):

    conv_weight_list = print_conv(model)
    filter_gl_list = torch.tensor([]).cuda()
    for name, w in conv_weight_list:
        ith_filter_reg_loss = torch.sum(torch.abs(w), dim=[1, 2, 3])
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return np.array(filter_gl_list.data.detach().cpu())


def print_conv(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'layer2.0.conv1' in name:
            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list
