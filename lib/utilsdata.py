#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:00:37 2019

@author: tianyu
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import Normalizer
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics.pairwise import euclidean_distances
import os
from sklearn import preprocessing
from sklearn import linear_model

def spilt_dataset(train_all_data, labels, shuffle_index):
        
    train_size, val_size = int(len(shuffle_index)* 0.8), int(len(shuffle_index)* 0.9)
    train_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(train_all_data).astype(np.float32)[shuffle_index[val_size:]]
    train_labels = labels[shuffle_index[0:train_size]]
    val_labels = labels[shuffle_index[train_size:val_size]]
    test_labels = labels[shuffle_index[val_size:]]
        
    ll, cnt = np.unique(train_labels,return_counts=True)
    return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
def generate_loader(train_data,val_data, test_data, train_labels, val_labels, test_labels, batchsize):
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)
    val_labels = val_labels.astype(np.int64)
    train_data = torch.FloatTensor(train_data)
    train_labels = torch.LongTensor(train_labels)
    val_data = torch.FloatTensor(val_data)
    val_labels = torch.LongTensor(val_labels)
    test_data = torch.FloatTensor(test_data)
    test_labels = torch.LongTensor(test_labels)
    
    dset_train = Data.TensorDataset(train_data, train_labels)
    train_loader = Data.DataLoader(dset_train, batch_size = batchsize, shuffle = True)
    dset_val = Data.TensorDataset(val_data, val_labels)
    val_loader = Data.DataLoader(dset_val, batch_size = len(dset_val), shuffle = False)
    dset_test = Data.TensorDataset(test_data, test_labels)
    test_loader = Data.DataLoader(dset_test, batch_size = len(dset_test), shuffle = False)
    
    return train_loader, val_loader, test_loader

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def high_var_npdata(data, num, gene = None, ind=False): #data: cell * gene
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 0)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data[:, gene_ind]
    if ind:
        return data[:,gene_ind],gene_ind
    return data[:,gene_ind],gene.iloc[gene_ind]


def high_tfIdf_npdata(data,tfIdf, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.var(tfIdf, axis = 0)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    np.random.shuffle(gene_ind)
    if gene is None and ind is False:
        return data[:,gene_ind]
    if ind:
        return data[:,gene_ind],gene_ind
    return data[:,gene_ind],gene.iloc[gene_ind]



def down_genes(alldata, adjall, num_gene):
    train_data_all = np.log1p(alldata)
    maxscale = np.max(train_data_all) 
    print('maxscale:',maxscale)
    train_data_all = train_data_all/np.max(train_data_all)    
    train_data_all, geneind = high_var_npdata(train_data_all, num= num_gene, ind=1)

    adj = adjall[geneind,:][:,geneind]
    adj = adj + sp.eye(adj.shape[0])
    
    print('load done.')
    adj = adj/np.max(adj)
    adj = adj.astype('float32')

    print('adj_shape:',adj.shape, ' [# cell, # gene]', train_data_all.shape)    
    return train_data_all, adj



def data_noise(data): # data is samples*genes
    for i in range(data.shape[0]):
        #drop_index = np.random.choice(train_data.shape[1], 500, replace=False)
        #train_data[i, drop_index] = 0
        target_dims = data.shape[1]
        noise = np.random.rand(target_dims)/10.0
        data[i] = data[i] + noise
    return data

def norm_max(data):        
    data = np.asarray(data)    
    max_data = np.max([np.absolute(np.min(data)), np.max(data)])
    data = data/max_data
    return data

def findDuplicated(df): 
    # input df: cells * genes
    df = df.T
    idx = df.index.str.upper()
    filter1 = idx.duplicated(keep = 'first')
    print('duplicated rows:',np.where(filter1 == True)[0])
    indd = np.where(filter1 == False)[0]
    df = df.iloc[indd]
    return df.T
    

# In[]:
def load_labels(path, dataset):
    
    labels = pd.read_csv(os.path.join(path + dataset) +'/Labels.csv',index_col = None)
    labels.columns = ['V1']
    class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['V1']))}
    labels['V1'] = labels['V1'].map(class_mapping)
    del class_mapping
    labels = np.asarray(labels).reshape(-1)  

    return labels
    

def load_largesc(path, dirAdj, dataset, net):
    
    if dataset == 'Zhengsorted':
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_DownSampled_SortedPBMC_data.csv',index_col = 0, header = 0)
        
    elif dataset == 'TM':
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_TM_data.csv',index_col = 0, header = 0)
        
    elif dataset == 'Xin':
        #path = os.path.join(path, 'Pancreatic_data/')
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_Xin_HumanPancreas_data.csv',index_col = 0, header = 0)
        
    elif dataset == 'BaronHuman':
        #path = os.path.join(path, 'Pancreatic_data/')
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_Baron_HumanPancreas_data.csv',index_col = 0, header = 0)
        
    elif dataset == 'BaronMouse':
        #path = os.path.join(path, 'Pancreatic_data/')
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_MousePancreas_data.csv',index_col = 0, header = 0)

    elif dataset == 'Muraro':
        #path = os.path.join(path, 'Pancreatic_data/')
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_Muraro_HumanPancreas_data_renameCols.csv',index_col = 0, header = 0)

    elif dataset == 'Segerstolpe':
        #path = os.path.join(path, 'Pancreatic_data/')   
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_Segerstolpe_HumanPancreas_data.csv',index_col = 0, header = 0)

    elif dataset == 'Zheng68K':
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_68K_PBMC_data.csv',index_col = 0, header = 0)
        
    elif dataset == '10x_5cl':        
        path = os.path.join(path, 'CellBench/') 
        features = pd.read_csv(os.path.join(path + dataset) +'/10x_5cl_data.csv',index_col = 0, header = 0)
    
    elif dataset == 'CelSeq2_5cl':        
        path = os.path.join(path, 'CellBench/')
        features = pd.read_csv(os.path.join(path + dataset) +'/CelSeq2_5cl_data.csv',index_col = 0, header = 0)
         
    features = findDuplicated(features)
    print(features.shape)
    adj = sp.load_npz(dirAdj + 'adj'+ net + dataset + '_'+str(features.T.shape[0])+'.npz')
    print(adj.shape)
    labels = load_labels(path, dataset)
    try:
        shuffle_index = np.loadtxt(os.path.join(path + dataset) +'/shuffle_index_'+dataset+'.txt')
    except OSError:
        shuffle_index = None
    
    return adj, np.asarray(features), labels,shuffle_index  # cells*genes







# In[]:
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels): # average of each batch 
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)#.requires_grad_()


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    #return img, pad_label, lens




       
        
