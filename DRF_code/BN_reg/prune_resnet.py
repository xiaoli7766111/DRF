import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from models import *
from Reg_tool import reg_total
from BN_reg.Main_tool import dataset, sparsity

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Prune settings
parser = argparse.ArgumentParser(description='prune')

parser.add_argument('--dataset', type=str, default='cifar10', help='training dataset ( data.cifar10 / mnist)')

parser.add_argument('--test-batch-size', type=int, default=256, help='input batch size for testing')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--depth', default=18, type=int, help='depth of network Vgg(19), Resnet(18/50/101)')

parser.add_argument('--model', default='NNN/r18xxx/04_33/Resnet_cifar10_100.model', type=str,
                    help='path to the model (default: none)')

parser.add_argument('--save', default='prune/r18xxx/04_33', type=str,
                    help='path to save pruned model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)


model = Resnet(depth=args.depth, dataset=args.dataset)

if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec = checkpoint['best_prec']

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))


defaultcfg = {
    'r18': [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512],
}

for name, param in model.named_parameters():
    if 'conv' in name:
        param.data = sparsity.zero_out(param, 0.0001).cuda()

filter_tap = reg_total.filter_gl(model)

cfg = defaultcfg['r18']
index = 0

# ---------------------------------预剪枝--------------------------------------
cfg_up = []
cfg_mask = []
for t in range(len(cfg)):
    rank_part = filter_tap[index: (index + cfg[t])]
    index += cfg[t]
    ak2 = (rank_part != 0).sum()
    cfg_mask.append(rank_part.clone())
    cfg_up.append(ak2.item())

print(f'Pre-processing Successful!')


def atest(model):

    _, test_loader = dataset.data_set(args.cuda, args.dataset, 64, args.test_batch_size)
    model.eval()
    correct = 0

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


if __name__ == '__main__':

    acc = atest(model)
    print("Cfg:", cfg_up)
    newmodel = Resnet(depth=args.depth, dataset=args.dataset, cfg=cfg_up)

    if args.cuda:
        newmodel.cuda()

    savepath = os.path.join(args.save, "prune.txt")
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n" + str(cfg_up) + "\n")
        fp.write("Test accuracy: \n" + str(acc) + "\n")

    old_modules = list(model.modules())
    new_modules = list(newmodel.modules())
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0
    chann = []
    numm = []
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        # print(m0)
        m1 = new_modules[layer_id]
        # print(m1)
        # bn 模块
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))

            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1    # 向后进行
            start_mask = end_mask.clone()

            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        # conv 模块
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))

                conv_count += 1
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                # print(0, conv_count)

            else:
                if args.depth == 18:   
                    if conv_count != 7 and conv_count != 12 and conv_count != 17:  
                        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
                        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))
                        # chann = idx0

                        if idx0.size == 1:
                            idx0 = np.resize(idx0, (1,))
                        if idx1.size == 1:
                            idx1 = np.resize(idx1, (1,))
                        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                        w1 = w1[idx1.tolist(), :, :, :].clone()
                        m1.weight.data = w1.clone()

                    else: 
                        # idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
                        # idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().detach().numpy())))
                        m1.weight.data = m0.weight.data.clone()

                    conv_count += 1
        # linear 模块
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().detach().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    torch.save({'cfg': cfg_up, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth'))
    print('ok')

    model = newmodel
    p_acc = atest(model)
    with open(savepath, "a") as fp:
        fp.write("\nPruned Test Accuracy: \n" + str(p_acc) + "\n")

