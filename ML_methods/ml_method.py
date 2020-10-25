import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from collections import Counter
#from imblearn.over_sampling import SMOTE
from sklearn import metrics

X_train = np.load('../data/segments_train_L_1024_errorL_20_mu2_3_x.npy')
y_train = np.load('../data/segments_train_L_1024_errorL_20_mu2_3_y.npy')

X_test = np.load('../data/segments_test_L_1024_errorL_20_mu2_3_x.npy')
y_test = np.load('../data/segments_test_L_1024_errorL_20_mu2_3_y.npy')

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#print('Original dataset shape %s' % Counter(y_train))
#
#sm = SMOTE(random_state=42)
#X_train, y_train = sm.fit_resample(X_train, y_train)

print('Resampled dataset shape %s' % Counter(y_train))

clf = LogisticRegression(C=1e5)
clf_svm = SVC(kernel='linear')
clf.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)
score = clf.score(X_test, y_test)
score_svm = clf_svm.score(X_test, y_test)

y_pred_lr = clf.predict(X_test)
y_pred_svm = clf_svm.predict(X_test)

print('LR test accuracy: {}, SVM linear test accuracy: {}'.format(score, score_svm))

conf_lr = confusion_matrix(y_test, y_pred_lr)
conf_svm = confusion_matrix(y_test, y_pred_svm)

print('LR confusion matrix: {}, \n svm confusion matrix: {}'.format(conf_lr, conf_svm))

y_score_lr = clf.decision_function(X_test)
y_score_svm = clf_svm.decision_function(X_test)

fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test, y_score_lr)
fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(y_test, y_score_svm)

AUC_lr = metrics.auc(fpr_lr, tpr_lr)
AUC_svm = metrics.auc(fpr_svm, tpr_svm)

print('LR AUC: {}, SVM AUC: {}'.format(AUC_lr, AUC_svm))
