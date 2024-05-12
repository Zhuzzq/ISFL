#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8

from audioop import cross
import os
import sys
import random
import time
import pickle
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from fed_op import average_weights, cal_weightings, cal_all_lips, param_deviation
from data_op import get_dataset, DatasetSplit, dataset_stat
from utils import train, test, model_init, lr_schedule, topk_test, args_parser
from collections import Counter


args = args_parser()


print(torch.__version__)
print(torch.version.cuda)
print(torch.version.cuda, torch.cuda.is_available())
device = torch.device(f'cuda:{args.GPU_idx}')
print(f"Using {device} device")


# fix random seed

if args.seed == -1:
    seed = random.randint(0, 23)
else:
    seed = args.seed


random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


if __name__ == '__main__':
    num_usr = args.num_usr
    global_epoch = args.global_epoch
    local_epoch = args.local_epoch
    dataset_name = args.dataset
    is_iid = args.iid
    sample_ratio = args.sample_ratio
    non_iid_ratio = args.noniid_ratio
    lips_size = args.lips_size
    sigma = args.sigma
    num_shard = args.num_shard
    if is_iid == 1:
        iid = 'iid'
    elif is_iid == 2:
        iid = 'mixniid'
    elif is_iid == 3:
        iid = 'Dniid'
    else:
        iid = 'noniid'

    batch_size = args.batch_size
    lr = 1e-3

    cl_num = 10
    if dataset_name == 'cifar100':
        cl_num = 100

    model_time = time.strftime("%m%d%H%M", time.localtime())
    print(f'model time{model_time}')
    start_time = time.time()

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(dataset_name, is_iid, num_usr, ratio=non_iid_ratio, alpha=non_iid_ratio, num_shard=num_shard)
    all_idxs = []
    [all_idxs.extend(user_groups[usr]) for usr in range(num_usr)]
    print(f'c_data volume: {len(all_idxs)}')
    c_dataset = DatasetSplit(dataset=train_dataset, idxs=all_idxs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    c_dataloader = DataLoader(c_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size * 2)
    print('total training batch num: ', len(train_dataloader))

    local_dataloaders = []
    local_datasets = []
    # local_labels = []
    usr_data_st = {}
    for usr in range(num_usr):
        local_datasets.append(DatasetSplit(dataset=train_dataset, idxs=user_groups[usr]))
        # local_labels.append(local_datasets[usr].dataset.targets)
        print(len(local_datasets[usr].labels))
        local_dataloaders.append(DataLoader(local_datasets[usr], batch_size=batch_size, shuffle=True))
        print(f'client-{usr}\ndata count: {len(local_datasets[usr])}\nbatch num: {len(local_dataloaders[usr])}')
        cls, cls_st = dataset_stat(local_datasets[usr], cl_num)
        usr_data_st[usr] = [cls, cls_st]
        print(f'cls: {cls}, stat: {cls_st}')

    # print(user_groups)

    # BUILD MODEL
    loss_fn = nn.CrossEntropyLoss()
    global_model, local_models, c_model = model_init(dataset_name, num_usr, lr, device)

    # Set the model to train and send it to device.
    print(global_model)
    print('# parameters:', sum(param.numel() for param in global_model.parameters()))

    # tensorboard
    log_dir = f'./logs/Sis-{dataset_name}-{iid}-{model_time}'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logger = SummaryWriter(log_dir)

    # Training
    train_loss, ta_accuracy = [], []
    test_loss, test_accuracy, test_accuracy5 = [], [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    lip_stat = []
    max_lips = []
    theta_dev = []
    cross_dev = []
    gds = []

    # w init
    ws = {}
    global_p = torch.tensor([usr_data_st[usr][1] for usr in range(num_usr)])
    global_p = torch.sum(global_p, dim=0)
    print('global p:')
    print(global_p)
    global_p = global_p / global_p.sum()
    print(global_p)
    for usr in range(num_usr):
        local_p = torch.tensor(usr_data_st[usr][1]) / torch.tensor(usr_data_st[usr][1]).sum()
        ws[usr] = torch.div(global_p, local_p)
        # ws[usr] = local_p
    w_records = [ws]



    dataset4lips = DatasetSplit(dataset=train_dataset, idxs=np.random.choice([i for i in range(len(train_dataset))], lips_size, replace=False))
    print(f'dataset for lips:{dataset_stat(dataset4lips,cl_num)}')

    for epoch in tqdm(range(global_epoch)):
        local_weights = []
        print(f'\n | Global Round : {epoch+1} |\n')
        print('Local training')

        for usr in range(num_usr):
            print(f'client-{usr} local train')

            # sampling
            if epoch > 0:
                samples_weight = np.array([w_records[-1][usr][int(y)] for y in local_datasets[usr].labels])
                # print(len(samples_weight))
                samples_weight = torch.tensor(samples_weight)
            else:
                samples_weight = np.array([w_records[0][usr][int(y)] for y in local_datasets[usr].labels])
                # print(len(samples_weight))
                samples_weight = torch.tensor(samples_weight)

            print(len(samples_weight))
            sampler = WeightedRandomSampler(samples_weight, num_samples=int(len(samples_weight) * sample_ratio))
            local_dataloaders[usr] = DataLoader(local_datasets[usr], batch_size=batch_size, sampler=sampler, drop_last=True)
            local_opt = torch.optim.Adam(local_models[usr].parameters(), lr)
            print(len(local_dataloaders[usr].dataset), len(local_dataloaders[usr]))
            for round in range(local_epoch):
                local_loss = train(local_dataloaders[usr], local_models[usr], loss_fn, local_opt, device)
                logger.add_scalars(f'local/train loss', {f'client-{usr}': local_loss}, round + 1 + local_epoch * epoch)

            # local test
            l_test_loss, l_test_acc, l_test_acc5 = topk_test(test_dataloader, local_models[usr], loss_fn, device)
            logger.add_scalars(f'local/test loss', {f'client-{usr}': l_test_loss}, epoch + 1)
            logger.add_scalars(f'local/test acc', {f'client-{usr}': l_test_acc}, epoch + 1)
            print(f"local-{usr} Test: \n Accuracy: {(l_test_acc):>0.1f}%, top-5 Accuracy: {(l_test_acc5):>0.1f}%, Loss: {l_test_loss:>8f} \n")

            local_weights.append(local_models[usr].state_dict())


        # cal lip
        usr_lip = {}
        ws = {}
        max_lip = []
        theta_d = []
        cross_d = []
        for usr in range(num_usr):
            print(f'client-{usr} weighting cal')
            # usr_lip[usr] = cal_all_lips(local_datasets[usr], device, local_models[usr], global_model, loss_fn, local_opts[usr], global_opt, int((1 - non_iid_ratio) * len(train_dataset)))
            usr_lip[usr] = cal_all_lips(dataset4lips, device, local_models[usr], global_model, loss_fn, cl=cl_num)
            lips = []

            for k in sorted(usr_lip[usr].keys()):
                lips.append(np.float32(max(usr_lip[usr][k])))
            max_lip.append(lips)
            # print(lips)
            res_w = cal_weightings(torch.tensor(lips), cl_num, global_p, torch.tensor(usr_data_st[usr][1]), sigma)
            ws[usr] = res_w
        w_records.append(ws)
        max_lips.append(max_lip)


        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        global_test_loss, test_acc, test_acc5 = topk_test(test_dataloader, global_model, loss_fn, device)

        test_loss.append(global_test_loss)
        test_accuracy.append(test_acc)
        test_accuracy5.append(test_acc5)

        logger.add_scalars(f'global/loss', {'test': global_test_loss}, epoch + 1)
        logger.add_scalars(f'global/acc', {'test': test_acc}, epoch + 1)


        # test on c-data
        global_c_test_loss, c_test_acc, c_test_acc5 = topk_test(c_dataloader, global_model, loss_fn, device)
        logger.add_scalars(f'global/c-loss', {'global-c-test': global_c_test_loss}, epoch + 1)
        logger.add_scalars(f'global/c-acc', {'global-c-test': c_test_acc}, epoch + 1)
        ta_accuracy.append([c_test_acc, c_test_acc5])

        # download weights
        for usr in range(num_usr):
            local_models[usr].load_state_dict(global_weights)
            # if dataset_name == 'cifar':
            #     local_schs[usr].step()

        print(f' \nTraining Stats after {epoch+1} global rounds:')
        # print(f'Training Loss : {global_train_loss}')
        # print('Train Accuracy: {:.2f}% \n'.format(train_acc))
        print(f'Test Loss : {global_test_loss}')
        print('Test Accuracy: {:.2f}% \n'.format(test_acc))
        print('Top5-Test Accuracy: {:.2f}% \n'.format(test_acc5))

    # Saving the objects train_loss and train_accuracy:

    # file_name = f'./logs/records/Sis_N{num_usr}_G{global_epoch}_L{local_epoch}_D{dataset_name}_iid{is_iid}-{non_iid_ratio}_B{batch_size}_S{seed}_L{lips_size}_sigma{sigma}_acc{max(test_accuracy)}_acc5{max(test_accuracy5)}-{model_time}.pkl'

    # with open(file_name, 'wb') as f:
    #     pickle.dump([usr_data_st, lip_stat, w_records, train_loss, ta_accuracy, test_loss, test_accuracy, max_lips, theta_dev, cross_dev, gds, test_accuracy5], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
