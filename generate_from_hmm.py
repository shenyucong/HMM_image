import numpy as np
from hmm import *
import matplotlib.pyplot as plt
import random

def generate_sample(n_samples, length, save_path):
    # normal = 0, error = 1
    transit = np.array([[0.9, 0.1],
                         [0.6, 0.4]])

    emission_mean = np.array([1.0, 2.0])
    emission_var = np.array([1.0, 1.0])

    startprob = np.array([1.0, 0.0])

    X, y = hmm_generate_data(transit, emission_mean, emission_var, startprob, n_samples, length)
    print(y)
    print(X.shape)
    print(y.shape)

    label = []
    for i in range(n_samples):
        if np.sum(y[i,:]) !=0:
            label.append(1)

        else:
            label.append(0)

    label = np.array(label)

    count1 = 0
    count2 = 0
    for i in range(label.shape[0]):
        if label[i] == 0:
            count1 += 1
        else:
            count2 +=1

    print('label 0: {}'.format(count1))
    print('label 1: {}'.format(count2))

    save_path_x = str(save_path) + '_x.npy'
    save_path_y = str(save_path) + '_y.npy'

    np.save(save_path_x, X)
    np.save(save_path_y, label)

    lst_0 = []
    lst_1 = []

    for i in range(label.shape[0]):
        if label[i] == 0:
            lst_0.append(i)
        else:
            lst_1.append(i)

    plot_0 = random.choices(lst_0, k=3)
    plot_1 = random.choices(lst_1, k=3)

    x = np.arange(0, length, 1)
    mu1 = np.ones((length,))
    mu2 = 2*np.ones((length,))


    cdict = {0: 'blue', 1: 'red'}
    fig, ax = plt.subplots(4, 1)
    group1 = y[plot_0[0]]
    for g in np.unique(group1):
        ix = np.where(group1 == g)
        ax[0].scatter(x, X[plot_0[0], :][ix], c = cdict[g], label = g, s = 100)
    ax[0].plot(x, mu1, label = 'class 0: mu=1', color='blue')
    ax[0].plot(x, mu2, label = 'class 1: mu=2', color='red')
    ax[0].legend()
    ax[0].set_title('Sample from class 0')

    group1 = y[plot_0[1]]
    for g in np.unique(group1):
        ix = np.where(group1 == g)
        ax[1].scatter(x, X[plot_0[1], :][ix], c = cdict[g], label = g, s = 100)
    ax[1].plot(x, mu1, label = 'class 0: mu=1', color='blue')
    ax[1].plot(x, mu2, label = 'class 1: mu=2', color='red')
    ax[1].legend()
    ax[1].set_title('Sample from class 0')

    group1 = y[plot_1[0]]
    for g in np.unique(group1):
        ix = np.where(group1 == g)
        x
        yy=X[plot_1[0], :][ix]
        ax[2].scatter(x[ix], X[plot_1[0], :][ix], c = cdict[g], label = g, s = 100)
    ax[2].plot(x, mu1, label = 'class 0: mu=1', color='blue')
    ax[2].plot(x, mu2, label = 'class 1: mu=2', color='red')
    ax[2].legend()
    ax[2].set_title('Sample from class 1')

    group1 = y[plot_1[2]]
    for g in np.unique(group1):
        ix = np.where(group1 == g)
        ax[3].scatter(x[ix], X[plot_1[2], :][ix], c = cdict[g], label = g, s = 100)
    ax[3].plot(x, mu1, label = 'class 0: mu=1', color='blue')
    ax[3].plot(x, mu2, label = 'class 1: mu=2', color='red')
    ax[3].legend()
    ax[3].set_title('Sample from class 1')

    plt.tight_layout()
    plt.show()


length = 49
train_path = './data/train_mu2_2_l_' + str(length)
test_path = './data/test__mu2_2_l_' + str(length)

generate_sample(10000, length, train_path)
generate_sample(2000, length, test_path)

