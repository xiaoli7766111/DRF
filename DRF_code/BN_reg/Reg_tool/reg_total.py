import torch


# 将 weight 存储 比每次循环节省时间
def get_conv_weight(model):
    conv_weight_list = []

    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:  # 仅计算卷积权重的稀疏
            weight = (name, param)
            conv_weight_list.append(weight)

    return conv_weight_list


# 存储 bn层 weight
def get_bn_weight(model):
    bn_weight_list = []

    for name, param in model.named_parameters():
        if 'bn' in name and 'weight' in name:
            print(name)
            weight = (name, param)
            bn_weight_list.append(weight)

    return bn_weight_list


# 计算filter的group lasso
def filter_gl(model):
    conv_weight_list = get_conv_weight(model)  # 得到模型参数
    filter_gl_list = torch.tensor([]).cuda()  # torch 列表操作

    for name, wight in conv_weight_list:
        ith_filter_reg_loss = torch.sqrt(torch.sum(torch.pow(wight, 2), dim=[1, 2, 3]))  # 进行GL平方和开根号
        # torch 添加list
        filter_gl_list = torch.cat([filter_gl_list, ith_filter_reg_loss], dim=0)

    return filter_gl_list


# 变动系数
def bn_change_loss(model, thr):

    num = bn_weight_up(model, thr)
    gl = filter_gl(model)  # 获取 filter 的 gl
    loss_tol = sum(torch.mul(num, gl))  # 进行 对应位置相乘

    return loss_tol


# 计算bn 层所有参数的和
def bn_tol_weight(model):

    bn_weight_list = get_bn_weight(model)
    total = 0
    for name, w in bn_weight_list:
        total += torch.sum(w)

    return total


# cpu 版
def bn_weight(model, thr):
    tol_weight = bn_tol_weight(model)
    weight_list = []
    for name, param in model.named_parameters():
        if 'bn' in name and 'weight' in name:
            weight = torch.mul(torch.sub(1, torch.div(param, tol_weight)), thr).cpu().detach().numpy()
            weight_list.extend(weight)

    return weight_list


# gpu 版
def bn_weight_up(model, thr):
    tol_weight = bn_tol_weight(model)
    weight_list2 = torch.tensor([]).cuda()
    for name, param in model.named_parameters():
        if 'bn' in name and 'weight' in name:
            weight = torch.mul(torch.sub(1, torch.div(param, tol_weight)), thr)
            weight_list2 = torch.cat([weight_list2, weight], dim=0)

    # a = weight_list2.detach().cpu().clone()
    # print(np.array(a))
    return weight_list2
