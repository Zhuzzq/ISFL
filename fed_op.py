#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

from collections import defaultdict
import copy
from threading import local
import torch
from torchvision import datasets
from torch import nn


def param_deviation(param1, param2):
    x = 0
    for k in range(len(param1)):
        x += torch.square(torch.linalg.norm(param1[k] - param2[k]))
        # print(param1[k],param2[k])
        # print(x)
    return torch.sqrt(x)


def grad_norm(param):
    x = 0
    for k in range(len(param)):
        x += torch.square(torch.linalg.norm(param[k].grad))
    return float(torch.sqrt(x))


def cal_gn_weights(chosen_data, device, model, loss_fn):
    gn = []

    print(f'data volume for gn_weights: {len(chosen_data)}')
    for (X, y) in chosen_data:
        X, y = X.to(device).unsqueeze(0), y.to(device).unsqueeze(0)

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        param = list(model.parameters())
        gn.append(grad_norm(param))
        model.zero_grad()
    return gn


def grad_deviation(param1, param2):
    x = 0
    for k in range(len(param1)):
        x += torch.square(torch.linalg.norm(param1[k].grad - param2[k].grad))
    return torch.sqrt(x)


def cal_Lip(param1, param2):
    if param_deviation(param1, param2) != 0:
        return float(grad_deviation(param1, param2) / param_deviation(param1, param2))
    else:
        print('0 param deviation!')
        return 0


def new_cal_lips(params0, params1):
    params = []
    grads = []
    for k in range(len(params0)):
        params.append(params0[k].view(-1) - params1[k].view(-1))
        grads.append(params0[k].grad.view(-1) - params1[k].grad.view(-1))
    params_dev = torch.linalg.norm(torch.cat(params))
    del params
    grads_dev = torch.linalg.norm(torch.cat(grads))
    # print(params_dev.size(), torch.linalg.norm(params_dev), torch.linalg.norm(grads_dev))
    del grads
    return grads_dev / params_dev


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def cal_all_lips(chosen_data, device, model0, model1, loss_fn, count=0, cl=10):
    lip_dict = defaultdict(list)

    print(f'data volume for callip: {len(chosen_data)}')

    k = 0
    for (X, y) in chosen_data:
        X, y = X.to(device).unsqueeze(0), y.to(device).unsqueeze(0)

        # model 1
        pred = model0(X)
        loss = loss_fn(pred, y)
        loss.backward()
        # Gradient clipping
        # if grad_clip:
        #     nn.utils.clip_grad_value_(model0.parameters(), grad_clip)
        param0 = list(model0.parameters())

        # model 2
        pred = model1(X)
        loss = loss_fn(pred, y)
        loss.backward()
        # if grad_clip:
        #     nn.utils.clip_grad_value_(model1.parameters(), grad_clip)
        param1 = list(model1.parameters())

        lip = cal_Lip(param0, param1)
        # lip = new_cal_lips(param0, param1)
        # print('label:',y[0],'\nLip:',lip)
        lip_dict[int(y[0])].append(lip)
        model0.zero_grad()
        model1.zero_grad()
        # optimizer0.zero_grad()
        # optimizer1.zero_grad()
        del X, y, pred, loss, param0, param1, lip
        k += 1
        if count and k >= count:
            break
    return lip_dict


def cal_weightings(lips, num_cls, global_p, local_p, sigma):
    local_p = local_p / local_p.sum()
    min_p = local_p * sigma
    CL = 1 - num_cls * torch.div(torch.square(lips), torch.square(lips).sum())
    alpha = torch.div(CL, torch.linalg.norm(CL))
    print(f'Lips: {lips}\n global p:{global_p}\n CL:{CL}\n alpha:{alpha}\nlocal p:{local_p}')
    alt_Gamma = torch.div(global_p - min_p, -alpha)
    Gamma = max(torch.max(alt_Gamma), 0)
    for g in alt_Gamma:
        if g <= 0:
            continue
        elif g < Gamma:
            Gamma = g
    q = global_p + Gamma * alpha
    # alt_Gamma = torch.div(global_p - min_p, alpha)
    # Gamma = max(torch.max(alt_Gamma), 0)
    # for g in alt_Gamma:
    #     if g <= 0:
    #         continue
    #     elif g < Gamma:
    #         Gamma = g
    # q = global_p - Gamma * alpha
    print(f'Gamma: {Gamma}\n Reweighted probability: {q}\n Weightings: {torch.div(q, local_p)}')
    # return q
    return torch.div(q, local_p)
