import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(r, r_1, i, i_1, s, device):
    inv_loss = invariance_loss(r, r_1)
    exp_loss = exposure_loss(r, device)
    tv_loss_i = tv_loss(i) + tv_loss(i_1)
    spa_loss = spatial_loss(r, s)

    loss = inv_loss + exp_loss + spa_loss + 10 * tv_loss_i

    return loss


def invariance_loss(r, r1):
    loss_function = nn.MSELoss()
    loss = loss_function(r, r1)
    return loss


def exposure_loss(r, device):
    loss_function = nn.MSELoss()
    r = F.avg_pool2d(r, kernel_size=16, stride=16)
    mask = torch.ones(r.size()).to(device)
    exp = 0.7
    mask = mask * exp
    loss = loss_function(r, mask)
    return loss


def spatial_loss(r, s):
    loss_function = nn.MSELoss()

    r = F.avg_pool2d(r, kernel_size=4, stride=4)
    s = F.avg_pool2d(s, kernel_size=4, stride=4)

    r_grad_h = calc_gradient(r, "h")
    r_grad_w = calc_gradient(r, "w")

    s_grad_h = calc_gradient(s, "h")
    s_grad_w = calc_gradient(s, "w")

    loss_h = loss_function(r_grad_h, s_grad_h)
    loss_w = loss_function(r_grad_w, s_grad_w)

    return loss_h + loss_w


def tv_loss(x):
    batch_size = x.size()[0]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((calc_gradient(x, "h")), 2).sum()
    w_tv = torch.pow((calc_gradient(x, "w")), 2).sum()
    loss = 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    return loss


def calc_gradient(x, direction):
    h_x = x.size()[2]
    w_x = x.size()[3]
    if direction == "h":
        return x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
    else:
        return x[:, :, :, 1:] - x[:, :, :, :w_x - 1]


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def disturbance(x):
    batch_size = x.size()[0]
    for i in range(batch_size):
        mean = torch.mean(x[i]).numpy()
        if mean <= 0.5:
            para = torch.rand(1)
        else:
            para = 4 * torch.rand(1) + 1

        x[i] = x[i] ** para

    return x
