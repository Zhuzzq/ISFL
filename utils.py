#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

import copy
from math import gamma
from threading import local
import torch
from torchvision import datasets
from nn_models import SMLP, MLP_3, CNN4mnist, MultiCNN4fmnist, CNNFashion_Mnist, Res18, CNNCifar, ResNet8, oriResNet9, ResNet8_moon, oriResNet9_moon, ResNet9, ResNet9_moon, EfficientNetB0
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_idx', '-GPU', type=int, default=7)
    parser.add_argument('--num_usr', '-N', type=int, default=10)
    parser.add_argument('--global_epoch', '-G', type=int, default=25)
    parser.add_argument('--local_epoch', '-L', type=int, default=5)
    parser.add_argument('--dataset', '-D', type=str, default='cifar')
    parser.add_argument('--iid', '-iid', type=int, default=1)
    parser.add_argument('--sample_ratio', '-sr', type=float, default=1.0)
    parser.add_argument('--noniid_ratio', '-nr', type=float, default=0.5)
    parser.add_argument('--lips_size', '-ls', type=int, default=1000)
    parser.add_argument('--sigma', '-s', type=float, default=0.02)
    parser.add_argument('--mu', '-mu', type=float, default=0.1)
    parser.add_argument('--tau', '-tau', type=float, default=0.05)
    parser.add_argument('--seed', '-seed', type=int, default=-1)
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--num_shard', '-ns', type=int, default=2)
    args = parser.parse_args()
    return args


def details(args):
    print('\nExperimental details:')
    print(f'Dataset            : {args.dataset}')
    if args.iid == 1:
        print('IID')
    elif args.iid == 0:
        print('Non-IID')
    elif args.iid == 2:
        print('M-NIIID')
    else:
        print('D-NIID')

    print(f'Optimizer          : {args.optimizer}')
    print(f'Global Rounds      : {args.round}')
    print(f'Local [E] Epoch    : {args.epoch}')
    print(f'Local [B] Batchsize: {args.batchsize}')
    print(f'Local [C] Fraction : {args.frac}')
    print(f'# Clients          : {args.num_users}\n')

    return


def train(dataloader, model, loss_fn, optimizer, device):
    # size = len(dataloader.dataset)
    # print(size)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # print(X.size(),y.size())
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        # print(batch, model.parameters())
        optimizer.step()
        optimizer.zero_grad()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct * 100


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        # batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs


def topk_test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, correct5 = 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            tmp = topk_accuracy(pred, y)
            correct += tmp[0].item()
            correct5 += tmp[1].item()
            # print(correct, correct5)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    correct5 /= size
    return test_loss, correct * 100, correct5 * 100


def model_init(dataset_name, num_usr, lr, device):
    local_models = []
    # local_opts = []
    if dataset_name == 'mnist':
        global_model = SMLP(512).to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(SMLP(512).to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))

    elif dataset_name == 'fmnist':
        global_model = CNNFashion_Mnist().to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        c_model = CNNFashion_Mnist().to(device)
        # c_opt = torch.optim.Adam(c_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(CNNFashion_Mnist().to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))

    elif dataset_name == 'cifar':
        global_model = ResNet8().to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        c_model = ResNet8().to(device)
        # c_opt = torch.optim.Adam(c_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(ResNet8().to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))

    elif dataset_name == 'cifar100':
        global_model = ResNet9(num_classes=100).to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        c_model = ResNet9(num_classes=100).to(device)
        # c_opt = torch.optim.Adam(c_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(ResNet9(num_classes=100).to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))

    return global_model, local_models, c_model


def model_init_moon(dataset_name, num_usr, lr, device):
    local_models = []
    # local_opts = []
    if dataset_name == 'cifar':
        global_model = ResNet8_moon().to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        c_model = ResNet8_moon().to(device)
        # c_opt = torch.optim.Adam(c_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(ResNet8_moon().to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))

    elif dataset_name == 'cifar100':
        global_model = ResNet9_moon(num_classes=100).to(device)
        # global_opt = torch.optim.Adam(global_model.parameters(), lr)
        c_model = ResNet9_moon(num_classes=100).to(device)
        # c_opt = torch.optim.Adam(c_model.parameters(), lr)
        for i in range(num_usr):
            local_models.append(ResNet9_moon(num_classes=100).to(device))
            # local_opts.append(torch.optim.Adam(local_models[i].parameters(), lr))
    return global_model, local_models, c_model


def lr_schedule(opt, step=8, decay=0.5):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=decay)
