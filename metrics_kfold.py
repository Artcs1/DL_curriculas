from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import torch

from torch.utils.data import Dataset

from Loader.emb_loader import Embedding


from tqdm import tqdm


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def metrics(Y, yhat):
    return accuracy_score(Y,yhat), recall_score(Y,yhat, average='macro'), precision_score(Y,yhat, average='macro'), f1_score(Y,yhat,average='macro')


def knn(X_train, y_train):
    knn_param = {'n_neighbors':[3,5,7]}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, knn_param)
    clf.fit(X_train,y_train)
    return clf

def log_reg(X_train, y_train):
    lg_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}
    lg = LogisticRegression()
    clf = GridSearchCV(lg, lg_param)
    clf.fit(X_train, y_train)
    return clf

def svm_linear(X_train, y_train):
    svm_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}
    svm = SVC(kernel='linear')
    clf = GridSearchCV(svm, svm_param)
    clf.fit(X_train, y_train)
    return clf

def svm_rbf(X_train, y_train):
    svm_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15],'gamma':[2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3]}
    svm = SVC(kernel='rbf')
    clf = GridSearchCV(svm, svm_param)
    clf.fit(X_train, y_train)
    return clf


def mlp(X_train, y_train):
    mlp_param = {'learning_rate_init':10**np.random.uniform(-3,-6,5),'alpha':10**np.random.uniform(-5,5,5)}
    mlp = MLPClassifier(max_iter=200)
    clf = GridSearchCV(mlp, mlp_param)
    clf.fit(X_train, y_train)
    return clf

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--nargs", nargs='+')
    args = parser.parse_args()

    acc = np.zeros((5,len(args.nargs)))
    rec = np.zeros((5,len(args.nargs)))
    pre = np.zeros((5,len(args.nargs)))
    f1s = np.zeros((5,len(args.nargs)))

    ind = 0
    for model in args.nargs:

        data = Embedding( model=model, sample='test')

        X = data.X.numpy()

        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)

        Y = data.Y.numpy()
        kf = StratifiedKFold(n_splits=4)

        KNN_L  = []
        LR_L   = []
        LSVM_L = []
        RSVM_L = []
        MLP_L  = []


        for train_index, test_index in tqdm(kf.split(X, Y)):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            KNN = knn(X_train, y_train)
            yhat = KNN.predict(X_test)
            KNN_m = metrics(y_test, yhat)
            KNN_L.append(KNN_m)

            LR = log_reg(X_train, y_train)
            yhat = LR.predict(X_test)
            LR_m = metrics(y_test, yhat)
            LR_L.append(LR_m)

            SVM = svm_linear(X_train, y_train)
            yhat = SVM.predict(X_test)
            SVM_m = metrics(y_test, yhat)
            LSVM_L.append(SVM_m)

            SVM = svm_rbf(X_train, y_train)
            yhat = SVM.predict(X_test)
            SVM_m = metrics(y_test, yhat)
            RSVM_L.append(SVM_m)

            MLP = mlp(X_train, y_train)
            yhat = MLP.predict(X_test)
            MLP_m = metrics(y_test, yhat)
            MLP_L.append(MLP_m)

        KNN_L  = np.mean(KNN_L,axis=0)
        LR_L   = np.mean(LR_L, axis=0)
        LSVM_L = np.mean(LSVM_L,axis=0)
        RSVM_L = np.mean(RSVM_L,axis=0)
        MLP_L  = np.mean(MLP_L,axis=0)

        acc[:,ind] = [KNN_L[0], LR_L[0], LSVM_L[0], RSVM_L[0], MLP_L[0]]
        rec[:,ind] = [KNN_L[1], LR_L[1], LSVM_L[1], RSVM_L[1], MLP_L[1]]
        pre[:,ind] = [KNN_L[2], LR_L[2], LSVM_L[2], RSVM_L[2], MLP_L[2]]
        f1s[:,ind] = [KNN_L[3], LR_L[3], LSVM_L[3], RSVM_L[3], MLP_L[3]]
        ind +=1

    iterables = [['ACC', 'RECALL','PRECISION', 'F1SCORE'], args.nargs]

    index = pd.MultiIndex.from_product(iterables)


    #df1 = pd.DataFrame(acc,columns=args.nargs, index=['KNN','LR','SVML','SVMR','MLP'])

    #df2 = pd.DataFrame(rec,columns=args.nargs, index=['KNN','LR','SVML','SVMR','MLP'])

    #df3 = pd.DataFrame(pre,columns=args.nargs, index=['KNN','LR','SVML','SVMR','MLP'])

    #df4 = pd.DataFrame(f1s,columns=args.nargs, index=['KNN','LR','SVML','SVMR','MLP'])


    out = pd.DataFrame(np.concatenate((acc,rec, pre, f1s),axis=1).round(3), columns=index , index=['KNN','LR','SVML','SVMR','MLP'])
    out.to_csv('results/metrics_kfold.csv')



if __name__ == '__main__':
    main()
