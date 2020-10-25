import numpy as np
import random

def hmm_generate_data(transit, emssion_mean, emssion_var, startprob, n_samples, length):
    '''
    Generate observed sequences by Hidden Markov Model
    Input:
        transit: transition matrix, i.e. must by square matrix
        emission_mean: A numpy array contains means of each states' emission probability distribution. (num_states,)
        emission_var: A numpy array contains variances of each states' emission probability distribution. (num_states,)
        startprob: starting probability. (num_states,)
        n_samples: number of samples to be generated
        length: length of sequence
    return:
        X: observed sequences, (n_samples, n_features)
        y: states, (n_samples, n_features)
    '''

    assert transit.shape[0] == transit.shape[0]
    assert transit.shape[0] == emssion_mean.shape[0]
    assert emssion_mean.shape[0] == emssion_var.shape[0]
    assert startprob.shape[0] == emssion_var.shape[0]

    num_states = transit.shape[0]

    X = np.empty(shape=[n_samples, length])
    y = np.empty(shape=[n_samples, length])

    currstate = int(random.choices(population=np.arange(num_states), weights=startprob)[0])

    for i in range(n_samples):

        X_temp = []
        y_temp = []
        for j in range(length):
            y_temp.append(currstate)
            emission_distribution = np.random.normal(emssion_mean[currstate], emssion_var[currstate], 1)
            X_temp.append(emission_distribution)

            pop = np.arange(num_states)
            weigt = transit[currstate, :]
            currstate_ = random.choices(population=np.arange(num_states), weights=transit[currstate, :].reshape(-1,))

            currstate = int(currstate_[0])

        X_temp = np.array(X_temp).reshape(-1,)
        y_temp = np.array(y_temp).reshape(-1,)
        X[i, :] = X_temp
        y[i, :] = y_temp
        #X = np.concatenate([X, np.array(X_temp)], axis=0)
        #y = np.concatenate([y, np.array(y_temp)], axis=0)


    return X, y


