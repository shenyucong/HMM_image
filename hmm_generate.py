print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm
from hmm import *

#startprob = np.array([0.6, 0.3, 0.1, 0.0])

# normal 1.0, error 0.0
#startprob = np.array([1.0, 0.0])
startprob = np.array([0.0, 1.0])

# The transition matrix, note that there are no transitions possible
# between component 1 and 3
#transmat = np.array([[0.7, 0.2, 0.0, 0.1],
#                     [0.3, 0.5, 0.2, 0.0],
#                     [0.0, 0.3, 0.5, 0.2],
#                     [0.2, 0.0, 0.2, 0.6]])
transmat = np.array([[0.6, 0.4],
                     [0.1, 0.9]])


dimension = 50
# The means of each component
means1 = np.ones((1, dimension))
means2 = 2*np.ones((1, dimension))
means = np.concatenate([means1, means2], axis = 0)
print(means.shape)
# The covariance of each component
covars = np.tile(np.identity(dimension), (2, 1, 1))
print(covars.shape)

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

# Generate samples
X, Y = model.sample(100)
print(X)
print('dimension of features: {}'.format(X.shape))
print('labels: {}'.format(Y.shape))
print(Y)
print(X)

count1 = 0
count2 = 0
for i in range(Y.shape[0]):
    if Y[i] == 0:
        count1 += 1
    else:
        count2 +=1

print('label 0: {}'.format(count1))
print('label 1: {}'.format(count2))


#np.save('HMM_X.npy', X)
#np.save('HMM_Y.npy', Y)

