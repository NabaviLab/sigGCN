#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:00:37 2020

@author: tianyu
"""
   
import sys, os
os.chdir("/Users/tianyu/Desktop/sigGCN")
#os.chdir("/gpfs/sharedfs1/shn15004/tianyu/spectralGCN")
#os.chdir("/shared/nabavilab/tianyu/")
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import pdb #pdb.set_trace()
import collections
import argparse
import time
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle, resample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import pandas as pd
import sys
sys.path.insert(0, 'lib/')


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)

from grid_graph import grid_graph
from coarsening import coarsen, laplacian
from coarsening import lmax_L
from coarsening import perm_data
from coarsening import rescale_L
from layermodel import *
import utilsdata
from utilsdata import *
import warnings
warnings.filterwarnings("ignore")
#
#
# Directories.
parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default='personal', help="personal or hpc")
parser.add_argument('--dir_data', type = str, default = os.path.join('..', 'data', 'cancer'), help = 'Directory to store data.')
# Graphs.
parser.add_argument('--numner_edges', type = int, default = 8, help = 'Graph: minimum number of edges per vertex.')
parser.add_argument('--metric', type=str, default = 'euclidean', help='Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
parser.add_argument('--normalized_laplacian', type=bool, default = True, help='Graph Laplacian: normalized.')
parser.add_argument('--coarsening_levels', type=int, default = 4, help='Number of coarsened graphs.')
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--epochs', type=int, default = 30, help='# of epoch')
parser.add_argument('--batchsize', type=int, default = 64, help='# of genes')
parser.add_argument('--dataset', type=str, default='Zhengsorted', help="dataset")
parser.add_argument('--id1', type=str, default = '', help='test in pancreas')
parser.add_argument('--id2', type=str, default = '', help='test in pancreas')

parser.add_argument('--net', type=str, default='String', help="netWork")
parser.add_argument('--clustering', type=str, default='kmeans', help="Clustering method")
parser.add_argument('--thres', type=float, default = 0.1, help='# of epoch')
parser.add_argument('--dist', type=str, default='', help="dist type")
parser.add_argument('--sampling_rate', type=float, default = 1, help='# sampling rate of cells')

args = parser.parse_args()
#['Xin','BaronHuman','Muraro','Segerstolpe', 'BaronMouse']

# # Feature graph
t_start = time.process_time()


# Load data
generateRankgenes = 0

if args.dataset not in ['pancreas','CellBench','negative']:
    generateTrainTest = 1
    filepath = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset/' if args.user == 'personal' else '/gpfs/sharedfs1/shn15004/tianyu/scRNAseq_Benchmark_datasets/Intra-dataset/'
    pathnet ='/Users/publicuser/Google Drive/fasttext/cnn/TCGA_cnn/BIOGRID-ALL-3.5.169.tab2.txt'    
    
    print('load data...')    
    adjall, alldata,labels,shuffle_index = utilsdata.load_largesc(path = filepath, dataset=args.dataset, net='String')

    features = np.log1p(alldata)
    maxscale = np.max(features) 
    print('maxscale:',maxscale)
    features = features/np.max(features)    
    features, geneind = utilsdata.high_var_npdata(features, num= args.num_gene, ind=1)

    adj = adjall[geneind,:][:,geneind]
    adj = adj + sp.eye(adj.shape[0])
    
    print('load done.')
    adj_for_loss = adj.todense()
    adj = adj/np.max(adj)
    adj = adj.astype('float32')

    print('******************************',adj.shape, features.shape)    




#####################################################

if generateTrainTest:
    shuffle_index = shuffle_index.astype(np.int32)
        
    train_size, val_size = int(len(shuffle_index)* 0.8), int(len(shuffle_index)* 0.9)
    train_data = np.asarray(features.T).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(features.T).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(features.T).astype(np.float32)[shuffle_index[val_size:]]
    train_labels = labels[shuffle_index[0:train_size]]
    val_labels = labels[shuffle_index[train_size:val_size]]
    test_labels = labels[shuffle_index[val_size:]]
        
    ll, cnt = np.unique(train_labels,return_counts=True)
    
    nclass = len(np.unique(labels))
    



        
        
L = [laplacian(adj, normalized=True)]

train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)
adj_for_loss = torch.FloatTensor(adj_for_loss)

dset_train = Data.TensorDataset(train_data, train_labels)
train_loader = Data.DataLoader(dset_train, batch_size = args.batchsize, shuffle = True)
dset_test = Data.TensorDataset(test_data, test_labels)
test_loader = Data.DataLoader(dset_test, shuffle = False)



##Delete existing network if exists
try:
    del model
    print('Delete existing network\n')
except NameError:
    print('No existing network to delete\n')



# network parameters
D_g = train_data.shape[1] # features(genes)
CL1_F = 5
CL1_K = 5
FC1_F = 32
FC2_F = 0
NN_FC1 = 256
NN_FC2 = 32
out_dim = nclass

net_parameters = [D_g, CL1_F, CL1_K, FC1_F,FC2_F,NN_FC1, NN_FC2, out_dim]
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: 
            m.bias.data.fill_(0.0)
            
# instantiate the object net of the class
net = Graph_GCN(net_parameters)
net.apply(weight_init)

if torch.cuda.is_available():
    net.cuda()
print(net)


# Weights
L_net = list(net.parameters())


# learning parameters
dropout_value = 0.2
l2_regularization = 5e-4
batch_size = args.batchsize
num_epochs = args.epochs
train_size = train_data.shape[0]
nb_iter = int(num_epochs * train_size) // batch_size
print('num_epochs=',num_epochs,', train_size=',train_size,', nb_iter=',nb_iter)


# Optimizer
global_lr = args.lr
global_step = 0
decay = 0.95
decay_steps = train_size


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * pow( decay , float(global_step// decay_steps) )
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
#optimizer = optim.Adam(net.parameters(),lr= args.lr, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr= args.lr)
criterion = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

## Train   
net.train()
losses_train = []
acc_train = []


t_total_train = time.time()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    # update learning rate
    cur_lr = adjust_learning_rate(optimizer,epoch)
    
    # reset time
    t_start = time.time()

    # extract batches
    epoch_loss = 0.0
    epoch_acc = 0.0
    count = 0
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
        optimizer.zero_grad()   
        out_gae, out_hidden, output, out_adj = net(batch_x, dropout_value, L)
        
        loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
        acc_batch = utilsdata.accuracy(output, batch_y).item()
        
        loss_batch.backward()
        optimizer.step()
        
        count += 1
        epoch_loss += loss_batch.item()
        epoch_acc += acc_batch
        global_step += args.batchsize 
        
        # print
        if count % 1000 == 0: # print every x mini-batches
            print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (epoch + 1, count, loss_batch.item(), acc_batch))


    epoch_loss /= count
    epoch_acc /= count
    losses_train.append(epoch_loss) # Calculating the loss
    acc_train.append(epoch_acc) # Calculating the acc
    # print
    t_stop = time.time() - t_start
    print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
          (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
    print('training_time:',t_stop)

t_total_train = time.time() - t_total_train

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.decomposition import PCA
# Test set


def test_model(loader,num_classes):
    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([num_classes,num_classes])
    predictions = pd.DataFrame()
    y_true = []
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        out_gae, out_hidden, pred, out_adj = net(batch_x, dropout_value, L)
        
        test_acc += utilsdata.accuracy(pred, batch_y).item()
        count += 1
        y_true.append(batch_y.item())
        #y_pred.append(pred.max(1)[1].item())
        confusionGCN[batch_y.item(), pred.max(1)[1].item()] += 1
        px = pd.DataFrame(pred.detach().cpu().numpy())            
        predictions = pd.concat((predictions, px),0)
        
    preds_labels = np.argmax(np.asarray(predictions), 1)
    test_acc = test_acc/float(count)
    predictions.insert(0, 'trueLabels', y_true)

    return test_acc, confusionGCN, predictions, preds_labels


t_start_test = time.time()
test_acc,confusionGCN, predictions, preds_labels = test_model(test_loader, nclass)
t_stop_test = time.time() - t_start_test    
print('  accuracy(test) = %.3f %%, time= %.3f' % (test_acc, t_stop_test))

testPreds4save = pd.DataFrame(preds_labels,columns=['predLabels'])
testPreds4save.insert(0, 'trueLabels', list(predictions.iloc[:,0]))
aa = np.exp(np.asarray(predictions.iloc[:,1:]))
confusionGCN = pd.DataFrame(confusionGCN)
totlePath = '/Users/tianyu/Desktop/scRNAseq_Benchmark_datasets/Intra-dataset'
OutputDir = totlePath +'/' + args.dataset +'/numCells/output' + str(args.sampling_rate)
#    if args.dataset in ['pancreas','CellBench']:
#        testPreds4save.to_csv('newgcn_test_preds_'+ args.dataset+str(args.num_gene) +'_'+str(id1)+str(id2) +'_'+str(CL1_F)+str(CL1_K)+'_'+args.dist+'.csv')
#        predictions.to_csv('newgcn_testProbs_preds_'+ args.dataset+str(args.num_gene)+'_'+str(id1)+str(id2) +'_'+str(CL1_F)+str(CL1_K)+'_'+args.dist+'.csv')
#        confusionGCN.to_csv('newgcn_confuMat_'+ args.dataset + str(args.num_gene)+'_'+str(id1)+str(id2) +'_'+str(CL1_F)+str(CL1_K)+'_'+args.dist+'.csv')
#        np.savetxt('newgcn_train_time_'+args.dataset + str(args.num_gene)+'_'+str(id1)+str(id2) +'_'+str(CL1_F)+str(CL1_K)+'_'+args.dist+'.txt', [t_total_train])   
#        np.savetxt('newgcn_test_time_'+args.dataset + str(args.num_gene)+'_'+str(id1)+str(id2) +'_'+str(CL1_F)+str(CL1_K)+'_'+args.dist+'.txt', [t_stop_test]) 
#
#    else:
#        testPreds4save.to_csv(OutputDir+'/newgcn_test_preds_'+ args.dataset+ str(args.num_gene)+'_'+str(CL1_F)+str(CL1_K)+''+args.dist+'.csv')
#        predictions.to_csv(OutputDir+'/newgcn_testProbs_preds_'+ args.dataset+ str(args.num_gene)+'_'+str(CL1_F)+str(CL1_K)+'' +args.dist+'.csv')
#        confusionGCN.to_csv(OutputDir+'/newgcn_confuMat_'+ args.dataset+ str(args.num_gene)+'_'+str(CL1_F)+str(CL1_K)+'' +args.dist+'.csv')    
#        np.savetxt(OutputDir+'/newgcn_train_time_'+args.dataset + str(args.num_gene) +'_'+str(CL1_F)+str(CL1_K)+''+args.dist+'.txt', [t_total_train])   
#        np.savetxt(OutputDir+'/newgcn_test_time_'+args.dataset + str(args.num_gene) +'_'+str(CL1_F)+str(CL1_K)+''+args.dist+'.txt', [t_stop_test]) 


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

def calculation(pred_test, test_labels, method='GCN'):
    test_acc = metrics.accuracy_score(pred_test, test_labels)
    test_f1_macro = metrics.f1_score(pred_test, test_labels, average='macro')
    test_f1_micro = metrics.f1_score(pred_test, test_labels, average='micro')
    precision = metrics.precision_score(test_labels, pred_test, average='micro')
    recall = metrics.recall_score(test_labels, pred_test, average='micro')
    fpr, tpr, thresholds = metrics.roc_curve(test_labels, pred_test, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    
    
    print('method','test_acc','f1_test_macro','f1_test_micro','Testprecision','Testrecall','Testauc')
    print(method, test_acc, test_f1_macro, test_f1_micro, precision,recall,auc )

calculation(preds_labels, predictions.iloc[:,0])

 
