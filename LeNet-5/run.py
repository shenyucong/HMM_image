from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
#import onnx
import random
from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn import metrics

#data_train = MNIST('./data/mnist',
#                   download=True,
#                   transform=transforms.Compose([
#                       transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))
#data_test = MNIST('./data/mnist',
#                  train=False,
#                  download=True,
#                  transform=transforms.Compose([
#                      transforms.Resize((32, 32)),
#                      transforms.ToTensor()]))
#data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
#data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

X_train = np.load('../data/segments_train_L_1024_errorL_50_mu2_3_x.npy')
y_train = np.load('../data/segments_train_L_1024_errorL_50_mu2_3_y.npy')

X_test = np.load('../data/segments_test_L_1024_errorL_50_mu2_3_x.npy')
y_test = np.load('../data/segments_test_L_1024_errorL_50_mu2_3_y.npy')

print('Original dataset shape %s' % Counter(y_train))

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

data_train = TensorDataset(X_train, y_train)
data_train_loader = DataLoader(data_train, batch_size = 64, shuffle=True)

data_test = TensorDataset(X_test, y_test)
data_test_loader = DataLoader(data_test, batch_size = 64)

net = LeNet5()
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)

#plot_X_index = []
#label_1_index = []
#label_0_index = []
#for i in range(y_test.numpy().shape[0]):
#    if y_test[i] == 0:
#        label_0_index.append(i)
#
#    else:
#        label_1_index.append(i)
#
#label_1_index = random.choices(label_1_index, k=3)
#label_0_index = random.choices(label_0_index, k=3)
#
#plt.figure()
#plt.subplot(2, 2, 1)
#plt.imshow(X_train[label_1_index[0], :, :, :].reshape(height, width))
#
#plt.subplot(2, 2, 2)
#plt.imshow(X_train[label_1_index[1], :, :, :].reshape(height, width))
#
#plt.subplot(2, 2, 3)
#plt.imshow(X_train[label_0_index[0], :, :, :].reshape(height, width))
#
#plt.subplot(2, 2, 4)
#plt.imshow(X_train[label_0_index[1], :, :, :].reshape(height, width))
#
#plt.show()



def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    y_pred = []

    y_score = []
    y = []
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

        for i in range(len(output.detach())):
            probability = np.max(torch.exp(output.detach()).numpy()[i])
            y_score.append(probability)
            y.append(labels.cpu().numpy()[i])

        pred = pred.numpy()
        for i in range(len(pred)):
            y_pred.append(pred[i])

    y_pred = np.array(y_pred)
    conf = confusion_matrix(y_test, y_pred)

    y_score = np.array(y_score)
    y = np.array(y)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_score)
    AUC = metrics.auc(fpr, tpr)

    avg_loss /= len(data_test)
    print('-------------------------------------------')
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    print('confusion matrix: {}'.format(conf))
    print('AUC: {}'.format(AUC))
    print('-------------------------------------------')

    return float(total_correct) / len(data_test)

def train_and_test(epoch):
    train(epoch)
    accuracy = test()

    #dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    #torch.onnx.export(net, dummy_input, "lenet.onnx")

    #onnx_model = onnx.load("lenet.onnx")
    #onnx.checker.check_model(onnx_model)
    return accuracy


def main():
    acc_lst = []
    for e in range(1, 50):
        acc = train_and_test(e)
        acc_lst.append(acc)

    acc_np = np.array(acc_lst)
    print('Best test accuracy: {}'.format(np.max(acc_np)))


if __name__ == '__main__':
    main()
