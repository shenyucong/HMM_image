import numpy as np
import matplotlib.pyplot as plt
import random

def generate_data(n_samples, mean1, var1, mean2, var2, length, error_length):
    n1_samples = int(n_samples / 2)
    n2_samples = int(n_samples / 2)

    X1 = np.empty(shape=[n1_samples, length])
    y1 = np.empty(shape=[n1_samples, length])

    for i in range(n1_samples):
        X1[i, :] = np.random.normal(mean1, var1, length)
        y1[i, :] = np.zeros(length)

    X2 = np.empty(shape=[n2_samples, length])
    y2 = np.empty(shape=[n2_samples, length])

    normal_length = length - error_length

    for i in range(n2_samples):
        x2_normal = np.random.normal(mean1, var1, normal_length)
        y2_normal = np.zeros(normal_length)

        insert_indice = random.randint(0, normal_length)

        x2_insert = np.random.normal(mean2, var2, error_length)
        y2_insert = np.ones(error_length)

        X2[i, :] = np.insert(x2_normal, insert_indice, x2_insert)
        y2[i, :] = np.insert(y2_normal, insert_indice, y2_insert)

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return X, y

def generate_sample(n_samples, length, error_length, save_path, mean1, var1, mean2, var2):
    # normal = 0, error = 1


    X, y = generate_data(n_samples, mean1, var1, mean2, var2, length, error_length)

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

def generate_segmentation_sample(n_samples, length, error_length, save_path, mean1, var1, mean2, var2):

    X, y = generate_data(n_samples, mean1, var1, mean2, var2, length, error_length)

    print('------------------------------------')
    print(y[10, :].shape)
    print('------------------------------------')
    print(y.shape)
    print('------------------------------------')

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
    save_path_seg_y = str(save_path) + '_segmentation_label.npy'

    np.save(save_path_x, X)
    np.save(save_path_y, label)
    np.save(save_path_seg_y, y)



# normal state
mean1 = 1
var1 = 1

# error state
mean2 = 10
var2 = 1

length = 1024
error_length = 100

train_path = './data/segments_train_L_' + str(length) + '_errorL_' + str(error_length) + '_mu2_' + str(mean2)
test_path = './data/segments_test_L_' + str(length) + '_errorL_' + str(error_length) + '_mu2_' + str(mean2)

generate_segmentation_sample(10000, length, error_length, train_path, mean1, var1, mean2, var2)
generate_segmentation_sample(2000, length, error_length, test_path, mean1, var1, mean2, var2)
