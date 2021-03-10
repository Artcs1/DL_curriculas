import numpy as np
import torch
from os import scandir, getcwd
from os.path import abspath,isfile



from torch.utils.data import Dataset

def ls(ruta = getcwd()):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


def load_data(paths, lens):
    classx = []
    ind = 0
    for x in lens:
        for i in np.arange(x):
            classx.append(ind)
        ind+=1
    ind =0
    E = []
    L = []
    for i in paths:
        f = open(i,'r', errors = 'ignore')
        embedding = f.read()
        clasess   = classx[ind]
        E.append(embedding)
        L.append(clasess)
        ind+=1
        f.close()
    return E,L

def format_curriculas(curriculas):
    L = []
    if type(curriculas) == type('str'):
        curricula = []
        curricula.append(curriculas)
        curriculas = curricula

    for curricula in curriculas:
        C = []
        courses = curricula.split("/\n")
        for i in range(len(courses)):
            if courses[i] != '':
                c = courses[i].replace("\n"," ")
                c = c.replace("\t","")
                if c[-1] == '/':
                    c = c[:-1]
                C.append(c.strip())
        L.append(C)
    return L

def toembedding(data,model,siz,name):
    print(name)
    nd = np.empty((0,siz), int)
    E = np.zeros(siz)
    E2 = np.zeros(siz)
    c1 =0
    for i in data:
        c2 = 0
        for strings in i:
            if name == 'bert' or name == 'cl_bert' or name == 'lm_bert' or name == 'ml_bert' :
                lista = []
                lista.append(strings)
                E2 = model.encode_sentences(lista ,combine_strategy = "mean").reshape(-1)
            else:
                T = strings.split(" ")
                c1 = 0
                for curso in T:
                    c1 = c1 +1
                    try:
                        E2 = E2 + model[curso]
                    except:
                        np.random.seed(143)
                        E2 = E2 + np.random.rand(E2.shape[0])
                E2 = (E2)/(c1+ 0.00000001)
            E = E + E2
            c2 = c2 + 1
        E = (E)/(c2+ 0.00000001)
        nd = np.append(nd, np.array([E]), axis=0)
    return nd


class Curriculas(Dataset):

    def __init__(self, path='DATA_TG100', data='train'):
        CS_paths = ls("./"+path+"/CS")
        CE_paths = ls("./"+path+"/CE")
        IT_paths = ls("./"+path+"/IT")
        IS_paths = ls("./"+path+"/IS")
        SE_paths = ls("./"+path+"/SE")
        PE_paths = ls("./"+path+"/PERU")

        length  = [len(CS_paths), len(CE_paths), len(IT_paths), len(IS_paths), len(SE_paths)]
        len_all = [len(CS_paths), len(CE_paths), len(IT_paths), len(IS_paths), len(SE_paths), len(PE_paths)]


        len_train = [ int(np.round(val * 0.6)) for val in length ]
        len_val   = [ int(np.round(val * 0.2)) for val in length ]
        len_test  = [ int(length[i] - len_train[i] - len_val[i]) for i in np.arange(len(length))]

        train_paths = CS_paths[0:len_train[0]] + CE_paths[0:len_train[1]] + IT_paths[0:len_train[2]] + IS_paths[0:len_train[3]] + SE_paths[0:len_train[4]]
        val_paths   = CS_paths[len_train[0]:len_train[0]+len_val[0]] + CE_paths[len_train[1]:len_train[1]+len_val[1]] + IT_paths[len_train[2]:len_train[2]+len_val[2]] + IS_paths[len_train[3]:len_train[3]+len_val[3]] + SE_paths[len_train[4]:len_train[4]+len_val[4]]
        test_paths  = CS_paths[len_train[0]+len_val[0]:len_train[0]+len_val[0]+len_test[0]] +  CE_paths[len_train[1]+len_val[1]:len_train[1]+len_val[1]+len_test[1]] +  IT_paths[len_train[2]+len_val[2]:len_train[2]+len_val[2]+len_test[2]] +  IS_paths[len_train[3]+len_val[3]:len_train[3]+len_val[3]+len_test[3]] +  SE_paths[len_train[4]+len_val[4]:len_train[4]+len_val[4]+len_test[4]]
        all_paths = CS_paths + CE_paths + IT_paths + IS_paths + SE_paths + PE_paths

        if data == 'train':
            E, L = load_data(train_paths, len_train)
            names = train_paths

        if data == 'valid':
            E, L = load_data(val_paths, len_val)
            names = val_paths

        if data == 'test':
            E, L = load_data(test_paths, len_test)
            names = test_paths

        if data == 'all':
            E, L = load_data(all_paths, len_all)
            names = all_paths

        self.x = E
        self.y = L
        self.names = names
        self.len = len(L)


    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return self.len

    def get_names(self):
        return self.names
