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
#from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from os import listdir
from os.path import isfile, join

from hypopt import GridSearch

import torch

from torch.utils.data import Dataset

from Loader.emb_loader import Embedding

def metrics(Y, yhat):
    return accuracy_score(Y,yhat), recall_score(Y,yhat, average='macro'), precision_score(Y,yhat, average='macro'), f1_score(Y,yhat,average='macro')


def knn(X_train, y_train, X_valid, y_valid):
    knn_param = {'n_neighbors':[3,5,7]}
    knn = KNeighborsClassifier()
    clf = GridSearch(knn, knn_param)
    clf.fit(X_train,y_train,X_valid,y_valid)
    return clf

def log_reg(X_train, y_train, X_valid, y_valid):
    lg_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}
    lg = LogisticRegression()
    clf = GridSearch(lg, lg_param)
    clf.fit(X_train, y_train, X_valid, y_valid)
    return clf

def svm_linear(X_train, y_train, X_valid, y_valid):
    svm_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15]}
    svm = SVC(kernel='linear')
    clf = GridSearch(svm, svm_param)
    clf.fit(X_train, y_train, X_valid, y_valid)
    return clf

def svm_rbf(X_train, y_train, X_valid, y_valid):
    svm_param = {'C':[2**-5,2**-3,2**-1,2**1,2**3,2**5,2**7,2**9,2**11,2**13,2**15],'gamma':[2**-15,2**-13,2**-11,2**-9,2**-7,2**-5,2**-3,2**-1,2**1,2**3]}
    svm = SVC(kernel='rbf')
    clf = GridSearch(svm, svm_param)
    clf.fit(X_train, y_train, X_valid, y_valid)
    return clf



def mlp(X_train, y_train, X_valid, y_valid):
    mlp_param = [{'learning_rate_init':10**np.random.uniform(-3,-6,5),'alpha':10**np.random.uniform(-5,5,5)}]
    mlp = MLPClassifier(max_iter=200)
    clf = GridSearch(mlp, mlp_param, parallelize = False)
    clf.fit(X_train, y_train, X_valid, y_valid)
    return clf

def main():

    parser = argparse.ArgumentParser(description = 'Curriculas')
    parser.add_argument("--nargs", nargs='+')
    args = parser.parse_args()

    acc = np.zeros((4,len(args.nargs)))
    rec = np.zeros((4,len(args.nargs)))
    pre = np.zeros((4,len(args.nargs)))
    f1s = np.zeros((4,len(args.nargs)))

    ind = 0

    for models_name in args.nargs:
        mypath = './Embeddings/'
        print(models_name)
        models = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith(models_name)]
        print(models)

        KNN_L  = []
        LR_L   = []
        LSVM_L = []
        RSVM_L = []
        MLP_L  = []


        for model in models:
            model = model[:-4]
            train_data = Embedding(model=model, sample='train')
            valid_data = Embedding(model=model, sample='valid')
            test_data  = Embedding(model=model, sample='test')

            X_train = train_data.X.numpy()
            print(X_train.shape)
            #scaler = preprocessing.StandardScaler().fit(X)
            scaler = preprocessing.Normalizer().fit(X_train)

            X_train = scaler.transform(X_train)
            Y_train = train_data.Y.numpy()

            X_valid = scaler.transform(valid_data.X.numpy())
            Y_valid = valid_data.Y.numpy()

            X_test  = scaler.transform(test_data.X.numpy())
            Y_test  = test_data.Y.numpy()

            KNN = knn(X_train, Y_train, X_valid, Y_valid)
            yhat = KNN.predict(X_test)
            print(yhat.shape)
            KNN_m = metrics(Y_test, yhat)
            KNN_L.append(KNN_m)

            LR = log_reg(X_train, Y_train, X_valid, Y_valid)
            yhat = LR.predict(X_test)
            LR_m = metrics(Y_test, yhat)
            LR_L.append(LR_m)

            SVM = svm_linear(X_train, Y_train, X_valid, Y_valid)
            yhat = SVM.predict(X_test)
            SVM_m = metrics(Y_test, yhat)
            LSVM_L.append(SVM_m)

            SVM = svm_rbf(X_train, Y_train, X_valid, Y_valid)
            yhat = SVM.predict(X_test)
            SVM_m = metrics(Y_test, yhat)
            RSVM_L.append(SVM_m)

        #MLP = mlp(X_train, Y_train, X_valid, Y_valid)
        #yhat = MLP.predict(X_test)
        #MLP_m = metrics(Y_test, yhat)
        #MLP_L.append(MLP_m)

        KNN_L  = np.mean(KNN_L,axis=0)
        LR_L   = np.mean(LR_L, axis=0)
        LSVM_L = np.mean(LSVM_L,axis=0)
        RSVM_L = np.mean(RSVM_L,axis=0)
        #MLP_L  = np.mean(MLP_L,axis=0)

        acc[:,ind] = [KNN_L[0], LR_L[0], LSVM_L[0], RSVM_L[0]]#, MLP_L[0]]
        rec[:,ind] = [KNN_L[1], LR_L[1], LSVM_L[1], RSVM_L[1]]#, MLP_L[1]]
        pre[:,ind] = [KNN_L[2], LR_L[2], LSVM_L[2], RSVM_L[2]]#, MLP_L[2]]
        f1s[:,ind] = [KNN_L[3], LR_L[3], LSVM_L[3], RSVM_L[3]]#, MLP_L[3]]
        ind +=1

    iterables = [['ACC', 'RECALL','PRECISION', 'F1SCORE'], args.nargs]

    index = pd.MultiIndex.from_product(iterables)

    out = pd.DataFrame(np.concatenate((acc,rec, pre, f1s),axis=1).round(3), columns=index , index=['KNN','LR','SVML','SVMR'])
    print(out)
    print(Y_test)
    out.to_csv('results/metrics_valid.csv')



if __name__ == '__main__':
    main()
