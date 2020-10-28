# -*- coding: utf-8 -*-

from __future__ import print_function
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, TensorDataset

from model.fcn import FCNs
#from Cityscapes_loader import CityscapesDataset
#from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os

from collections import Counter
import scipy.misc
from sklearn.model_selection import train_test_split

n_class    = 2

batch_size = 6
epochs     = 500
lr         = 1e-4
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

#if sys.argv[1] == 'CamVid':
#    root_dir   = "CamVid/"
#else:
#    root_dir   = "CityScapes/"
#train_file = os.path.join(root_dir, "train.csv")
#val_file   = os.path.join(root_dir, "val.csv")

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

X_train = np.load('../../data/segments_train_L_1024_errorL_100_mu2_10_x.npy')
y_train = np.load('../../data/segments_train_L_1024_errorL_100_mu2_10_segmentation_label.npy')

X_test = np.load('../../data/segments_test_L_1024_errorL_100_mu2_10_x.npy')
y_test = np.load('../../data/segments_test_L_1024_errorL_100_mu2_10_segmentation_label.npy')

#print('Original dataset shape %s' % Counter(y_train))

height = 32
width = 32

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X_train, y_train)
#
#print('Resampled dataset shape %s' % Counter(y_train))

print('train data shape: {}'.format(X_train.shape))
print('test data shaep {}'.format(X_test.shape))

X_train = np.reshape(X_train, (-1, 1, height, width))
X_test = np.reshape(X_test, (-1, 1, height, width))

X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

#vgg_model = VGGNet(requires_grad=True, remove_fc=True)
#fcn_model = FCN(pretrained_net=None, n_class=n_class)
fcn_model = FCNs()

if use_gpu:
    ts = time.time()
    #vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iter, data in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:
                inputs = Variable(data[0].cuda())
                labels = Variable(data[1].cuda())
            else:
                inputs, labels = Variable(data[0]), Variable(data[1])

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)

        val(epoch)


def val(epoch):
    fcn_model.eval()
    total_ious = []
    pixel_accs = []
    #for iter, batch in enumerate(val_loader):
    for iter, data in enumerate(val_loader):
        if use_gpu:
            #inputs = Variable(batch['X'].cuda())
            inputs = Variable(data[0].cuda())
        else:
            inputs = Variable(batch['X'])

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        print(h, w, N)
        print(output.shape)
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        #target = batch['l'].cpu().numpy().reshape(N, h, w)
        target = data[1].cpu().numpy().reshape(N, h, W)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()
