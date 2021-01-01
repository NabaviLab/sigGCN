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

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot



def high_var_dfdata_gene(data, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar) #small --> big
    if gene is None and ind is False:
        return data.iloc[ind_maxvar[:num]]
    if ind:
        return data.iloc[ind_maxvar[:num]], ind_maxvar[:num]
    ind_gene = data.index.values[ind_maxvar[:num]]
    return data.iloc[ind_maxvar[:num]],gene.loc[ind_gene]

def high_var_dfdata(data, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data.iloc[ind_maxvar[:num]]
    if ind:
        return data.iloc[gene_ind], gene_ind
    return data.iloc[gene_ind],gene.iloc[gene_ind]

def high_var_npdata(data, num, gene = None, ind=False): #data: gene*cell
    dat = np.asarray(data)
    datavar = np.var(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data[gene_ind]
    if ind:
        return data[gene_ind],gene_ind
    return data[gene_ind],gene.iloc[gene_ind]


def high_tfIdf_npdata(data,tfIdf, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.var(tfIdf, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    np.random.shuffle(gene_ind)
    if gene is None and ind is False:
        return data[gene_ind]
    if ind:
        return data[gene_ind],gene_ind
    return data[gene_ind],gene.iloc[gene_ind]

def high_expr_dfdata(data, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.sum(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
    if gene is None and ind is False:
        return data.iloc[gene_ind]
    if ind:
        return data.iloc[gene_ind], gene_ind
    return data.iloc[gene_ind],gene.iloc[gene_ind]

def high_expr_npdata(data, num, gene = None, ind=False):
    dat = np.asarray(data)
    datavar = np.sum(dat, axis = 1)*(-1)
    ind_maxvar = np.argsort(datavar)
    gene_ind = ind_maxvar[:num]
#    np.random.shuffle(gene_ind)
    if gene is None and ind is False:
        return data[gene_ind]
    if ind:
        return data[gene_ind],gene_ind
    return data[gene_ind],gene.iloc[gene_ind]

def get_rank_gene(OutputDir, dataset):
    gene = pd.read_csv(OutputDir+dataset+'/rank_genes_dropouts_'+dataset+'.csv')
    return gene
    
def rank_gene_dropouts(data, OutputDir, dataset):
    # data: n_cell * n_gene
    genes = np.zeros([np.shape(data)[1],1], dtype = '>U10')
    train = pd.DataFrame(data)
    train.columns = np.arange(len(train.columns))

    # rank genes training set
    dropout = (train == 0).sum(axis='rows') # n_gene * 1
    dropout = (dropout / train.shape[0]) * 100
    mean = train.mean(axis='rows') # n_gene * 1

    notzero = np.where((np.array(mean) > 0) & (np.array(dropout) > 0))[0] 
    zero = np.where(~((np.array(mean) > 0) & (np.array(dropout) > 0)))[0]
    train_notzero = train.iloc[:,notzero]
    train_zero = train.iloc[:,zero]
    zero_genes = train_zero.columns

    dropout = dropout.iloc[notzero]
    mean = mean.iloc[notzero]

    dropout = np.log2(np.array(dropout)).reshape(-1,1)
    mean = np.array(mean).reshape(-1,1)
    reg = linear_model.LinearRegression()
    reg.fit(mean,dropout)

    residuals = dropout - reg.predict(mean)
    residuals = pd.Series(np.array(residuals).ravel(),index=train_notzero.columns) # n_gene * 1
    residuals = residuals.sort_values(ascending=False)
    sorted_genes = residuals.index
    sorted_genes = sorted_genes.append(zero_genes)

    genes[:,0] = sorted_genes.values
    genes = pd.DataFrame(genes)
    genes.to_csv(OutputDir + dataset + "/rank_genes_dropouts_" + dataset + ".csv", index = False)



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
    

def load_largesc(path = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/', dataset='Zhengsorted',net='String'):
    
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

    elif dataset == 'AMB':
        features = pd.read_csv(os.path.join(path + dataset) +'/Filtered_mouse_allen_brain_data.csv',index_col = 0, header = 0)
        features = findDuplicated(features)
        print(features.shape)
        adj = sp.load_npz(os.path.join(path + dataset) + '/adj'+ net + dataset + '_'+str(features.T.shape[0])+'.npz')
        print(adj.shape)

        shuffle_index = np.loadtxt(os.path.join(path + dataset) +'/shuffle_index_'+dataset+'.txt')
        
        labels = pd.read_csv(os.path.join(path + dataset) +'/Labels.csv',index_col = None)
        class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['Class']))}
        labels['Class'] = labels['Class'].map(class_mapping)
        del class_mapping
        labels = np.asarray(labels.iloc[:,0]).reshape(-1) 
        
        return adj, np.asarray(features.T), labels,shuffle_index

    
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
    adj = sp.load_npz(os.path.join(path + dataset) + '/adj'+ net + dataset + '_'+str(features.T.shape[0])+'.npz')
    print(adj.shape)
    labels = load_labels(path, dataset)
    shuffle_index = np.loadtxt(os.path.join(path + dataset) +'/shuffle_index_'+dataset+'.txt')
    
    return adj, np.asarray(features.T), labels,shuffle_index



# In[]:
def load_pancreas(path = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/', dataset='',net='String'):

    ##############
    xin = pd.read_csv(os.path.join(path + 'Xin') +'/Filtered_Xin_HumanPancreas_data.csv',index_col = 0, header = 0)
    bh = pd.read_csv(os.path.join(path + 'BaronHuman') +'/Filtered_Baron_HumanPancreas_data.csv',index_col = 0, header = 0)
    mu = pd.read_csv(os.path.join(path + 'Muraro') +'/Filtered_Muraro_HumanPancreas_data_renameCols.csv',index_col = 0, header = 0)
    se = pd.read_csv(os.path.join(path + 'Segerstolpe') +'/Filtered_Segerstolpe_HumanPancreas_data.csv',index_col = 0, header = 0)
    
    gene_set = list(set(xin.columns)&set(bh.columns)&set(mu.columns)&set(se.columns))
    gene_set.sort()
    gene_index_bh = [i for i, e in enumerate(bh.columns) if e in gene_set]
    xin = xin[gene_set]
    bh = bh[gene_set]
    mu = mu[gene_set]
    se = se[gene_set]
    
    mu = np.log1p(mu)
    se = np.log1p(se)
    bh = np.log1p(bh)
    xin = np.log1p(xin)
#    indexXin = xin.index.to_list()
#    indexMu = mu.index.to_list()
#    indexSe = se.index.to_list()
#    indexBh = bh.index.to_list()
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = min_max_scaler.fit_transform(np.asarray(mu))
    mu = pd.DataFrame(temp, index = mu.index, columns = mu.columns)
    temp = min_max_scaler.fit_transform(np.asarray(se))
    se = pd.DataFrame(temp, index = se.index, columns = se.columns)
    temp = min_max_scaler.fit_transform(np.asarray(bh))
    bh = pd.DataFrame(temp, index = bh.index, columns = bh.columns)
    temp = min_max_scaler.fit_transform(np.asarray(xin))
    xin = pd.DataFrame(temp, index = xin.index, columns = xin.columns)
    del temp
    #mu = preprocessing.normalize(np.asarray(mu), axis = 1, norm='l1')
    
    
    
    ############### 
    features = pd.read_csv(os.path.join(path + 'BaronHuman') +'/Filtered_Baron_HumanPancreas_data.csv',index_col = 0, header = 0, nrows=2)      
    features = findDuplicated(features)
    print(features.shape)
    adj = sp.load_npz(os.path.join(path + 'BaronHuman') + '/adj'+ net + 'BaronHuman' + '_'+str(features.T.shape[0])+'.npz')
    print(adj.shape)
    adj = adj[gene_index_bh, :][:, gene_index_bh]
    
    ###############
    datasets = ['Xin','BaronHuman','Muraro','Segerstolpe', 'BaronMouse']
    l_xin = pd.read_csv(os.path.join(path + datasets[0]) +'/Labels.csv',index_col = None)
    l_bh = pd.read_csv(os.path.join(path + datasets[1]) +'/Labels.csv',index_col = None) 
    l_mu = pd.read_csv(os.path.join(path + datasets[2]) +'/Labels.csv',index_col = None) 
    l_mu = l_mu.replace('duct','ductal')
    l_mu = l_mu.replace('pp','gamma')
    l_se = pd.read_csv(os.path.join(path + datasets[3]) +'/Labels.csv',index_col = None) 
    #labels_set = list(set(l_xin['x']) & set(l_bh['x']) & set(l_mu['x']))
    
    if True:
        labels_set = set(['alpha','beta','delta','gamma'])
        index = [i for i in range(len(l_mu)) if l_mu['x'][i] in labels_set]
        mu = mu.iloc[index]
        l_mu = l_mu.iloc[index]
        index = [i for i in range(len(l_se)) if l_se['x'][i] in labels_set]
        se = se.iloc[index]
        l_se = l_se.iloc[index]
        index = [i for i in range(len(l_bh)) if l_bh['x'][i] in labels_set]
        bh = bh.iloc[index]
        l_bh = l_bh.iloc[index]
        index = [i for i in range(len(l_xin)) if l_xin['x'][i] in labels_set]
        xin = xin.iloc[index]
        l_xin = l_xin.iloc[index]
    alldata = pd.concat((xin,bh,mu,se), 0)

    #alldata.to_csv(path+'Data_pancreas_4.csv')
    
    labels = pd.concat((l_xin, l_bh, l_mu, l_se), 0)
#    labels.to_csv(path+'Labels_pancreas_19.csv')
    labels.columns = ['V1']
    class_mapping = {label: idx for idx, label in enumerate(np.unique(labels['V1']))}
    labels['V1'] = labels['V1'].map(class_mapping)
    del class_mapping
    labels = np.asarray(labels).reshape(-1)  
    ###############
    #shuffle_index = np.asarray([1449, 8569, 2122,2133])
    shuffle_index = np.asarray([1449, 5707, 1554, 1440])
    
    return adj, np.asarray(alldata.T), labels, shuffle_index    

# In[]:

def build_adj_weight(idx_features):

    edges_unordered =  pd.read_csv('/users/tianyu/desktop/imputation/STRING_ggi.csv', index_col = None, usecols = [1,2,16]) 
#    edges_unordered = np.asarray(edges_unordered[['protein1','protein2','combined_score']])   # Upper case.
    edges_unordered = np.asarray(edges_unordered) 
    
    idx = []
    mapped_index = idx_features.index.str.upper() # if data.index is lower case. Usoskin data is upper case, do not need it.
    for i in range(len(edges_unordered)):
        if edges_unordered[i,0] in mapped_index and edges_unordered[i,1] in mapped_index:
            idx.append(i)
    edges_unordered = edges_unordered[idx]
    print ('idx_num:',len(idx))
    del i,idx
    
    # build graph
    idx = np.array(mapped_index)
    idx_map = {j: i for i, j in enumerate(idx)} # eg: {'TSPAN12': 0, 'TSHZ1': 1}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    edges = np.array(list(map(idx_map.get, edges_unordered[:,0:2].flatten())),
                     dtype=np.int32).reshape(edges_unordered[:,0:2].shape) #map：map(function, element):function on element. 
    
    adj = sp.coo_matrix((edges_unordered[:, 2], (edges[:, 0], edges[:, 1])),
                    shape=(idx_features.shape[0], idx_features.shape[0]),
                    dtype=np.float32)
    #del idx,idx_map,edges_unordered
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = (adj + sp.eye(adj.shape[0])) #diagonal, set to 1

    
    return adj
    

def getAdjByBiogrid(idx_features, pathnet = '~/Google Drive/fasttext/cnn/TCGA_cnn/BIOGRID-ALL-3.5.169.tab2.txt'):
    edges_unordered =  pd.read_table(pathnet ,index_col=None, usecols = [7,8] )
    edges_unordered = np.asarray(edges_unordered)  
    
    idx = []
    for i in range(len(edges_unordered)):
        if edges_unordered[i,0] in idx_features.index and edges_unordered[i,1] in idx_features.index:
            idx.append(i)
    edges_unordered = edges_unordered[idx]
    del i,idx
    
    # build graph
    idx = np.array(idx_features.index)
    idx_map = {j: i for i, j in enumerate(idx)}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape) #map：map(function, element):function on element
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx_features.shape[0], idx_features.shape[0]),
                        dtype=np.float32)
    del idx,idx_map,edges_unordered
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#    adj = adj + sp.eye(adj.shape[0])
    
#    sp.save_npz(os.path.join(pathnet,'adjCancer18442.npz'), adj)
       
    return adj

def removeZeroAdj(adj, gedata):
    #feature size: genes * samples, numpy.darray 
    if adj[0,0] != 0:
        #adj = adj - sp.eye(adj.shape[0])
        adj.setdiag(0)
#    adjdense = adj.todense()
    indd = np.where(np.sum(adj, axis=1) != 0)[0]
    adj = adj[indd, :][:, indd]
#    adjdense = adjdense[indd,:]
#    adjdense = adjdense[:, indd]
    gedata = gedata[indd,:]

    
    return adj, gedata




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




       
        
