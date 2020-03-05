#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:53:39 2019

@author: qwang
"""


import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


#%%
class RoBDataset(Dataset):
    """ RoB dataset """
    
    def __init__(self, info_file, mat_dir, rob_item, max_len=None, group=None):  
        
        self.mat_dir = mat_dir
        self.rob_item = rob_item     
        self.max_len = max_len
        
        info_df = pd.read_pickle(info_file)
        if group:
            info_df = info_df[info_df['partition']==group]
        self.info_df = info_df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        """
        rob_item:
            'RandomizationTreatmentControl'
            'BlindedOutcomeAssessment'
            'SampleSizeCalculation'
            'AnimalExclusions'     
            'AllocationConcealment'
            'AnimalWelfareRegulations'
            'ConflictsOfInterest'
        """
        dmat_path = os.path.join(self.mat_dir, self.info_df.loc[idx, 'goldID']+'.pkl')  
        doc_df = pd.read_pickle(dmat_path)    
        
        # Cut/Pad doc mat
        if self.max_len < len(doc_df):
            doc_df = doc_df[:self.max_len]
        else:
            zero_df = pd.DataFrame(np.zeros((self.max_len-len(doc_df), 512)))
            doc_df = pd.concat([doc_df, zero_df])
        
        label = self.info_df.loc[idx, self.rob_item]
        
        doc_tensor = torch.tensor(doc_df.values).float()
        label_tensor = torch.tensor([label]).float()

        return doc_tensor, label_tensor
    
    def cls_weight(self):
        df = self.info_df   
        n_pos = len(df[df[self.rob_item]==1])
        n_neg = len(df[df[self.rob_item]==0])     
        return [1/n_pos, 1/n_neg]
    
        
class PadDoc:
    def __call__(self, batch):
        # Element in batch: (doc, label)
        # Sort batch by doc_len in descending order. x[0] => doc, x[1] -> label
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)  # 
        
		# Pad doc within batch
        docs = [x[0] for x in sorted_batch]
        docs_padded = nn.utils.rnn.pad_sequence(docs, batch_first=True)
        
		# Obtain labels of sorted batch
        labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
        
        # Store length of each doc for unpad them later
        lens = torch.LongTensor([len(x) for x in docs])  
        return docs_padded, labels, lens
        
#%% Instance   
#data_dir = '/media/mynewdrive/rob/'
#os.chdir(data_dir)
#train_set = RoBDataset(info_file='data/rob_info_a.pkl',
#                       mat_dir='data/rob_mat',
#                       rob_item='RandomizationTreatmentControl',
#                       group='train',
#                       max_len=100)
#
#len(train_set)  # 6272
#train_set.cls_weight()
#
#idx = 0
#doc, label = train_set[idx]
#doc.size()  # torch.Size([195, 512])
#label.size() # torch.Size([1])
#
#for i in range(len(train_set)):
#    doc, label = train_set[i]
#    print(label)
#    if i == 20: break
#
## DataLoader
#from torch.utils.data import DataLoader
#train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, collate_fn=PadDoc())
#
#
#batch = next(iter(train_loader))
#doc_batch = batch[0]; print(doc_batch)   
#label_batch = batch[1]; print(label_batch)    
#len_batch = batch[2]; print(len_batch)  
#
#doc_batch.size()  # [batch_size, doc_len, embed_dim]
#label_batch.size()  # [batch_size]
#len_batch.size()  # [batch_size]
#
#for i, rob_batch in enumerate(train_loader):
#    print("[batch {}] Doc: {}, Label: {}".format(i, rob_batch[0].size(), rob_batch[1].size()))
#    if i == 5: break



