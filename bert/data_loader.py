#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:15:51 2020
ref: https://github.com/AndriyMulyar/bert_document_classification/blob/master/bert_document_classification/document_bert.py

@author: qwang
"""
import os
import pickle
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


   
#%%
class DocDataset(Dataset):
    """
    info_file: 
        'rob_info_a.pkl' (for RandomizationTreatmentControl | BlindedOutcomeAssessment | SampleSizeCalculation | AnimalExclusions)     
        'rob_info_b.pkl' (for AllocationConcealment | AnimalWelfareRegulations | ConflictsOfInterest)   
    group: 'train', 'valid', 'test'  
    
    Returns:
        doc: [num_chunks, 3, max_chunk_len]. '3' refers to tokens_ids, attn_masks, token_type_ids
        label
        
    """
    def __init__(self, info_file, pkl_dir, rob_item, max_chunk_len, cut_head_ratio, cut_tail_ratio, group, tokenizer):  
        
        self.pkl_dir = pkl_dir
        self.rob_item = rob_item   
        self.max_chunk_len = max_chunk_len
        self.cut_head_ratio = cut_head_ratio
        self.cut_tail_ratio = cut_tail_ratio
        self.tokenizer = tokenizer
        
        info_df = pd.read_pickle(info_file)
        print('Overal data size: {}'.format(len(info_df)))
               
        if group:
            info_df = info_df[info_df['partition']==group]
        self.info_df = info_df.reset_index(drop=True)
        
        
    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        
        pkl_path = os.path.join(self.pkl_dir, self.info_df.loc[idx, 'goldID']+'.pkl') 
        with open(pkl_path, 'rb') as fin:
            sent_ls = pickle.load(fin)
        
        # Cut sentences from head/tail
        doc_len = len(sent_ls)
        n_head = int(doc_len * self.cut_head_ratio)
        n_tail = int(doc_len * self.cut_tail_ratio)
        if n_tail == 0:
            sent_ls = sent_ls[n_head:]
        else:
            sent_ls = sent_ls[n_head : -n_tail]
        
        # Split sentence lists to tokens
        tokens = []
        for l in sent_ls:
            tokens = tokens + l[0].split(" ")
            
        # Split tokens into chunks
        n_chunks = len(tokens) // (self.max_chunk_len - 2)
        if len(tokens) % (self.max_chunk_len - 2) != 0:
            n_chunks += 1           
        
        if n_chunks > 20:
            tokens = tokens[:self.max_chunk_len*20]
            n_chunks = 20
            
        assert n_chunks <= 20, "The document is too large. Try to increase cut_head/tail_ratio."
        
        # Document tensor
        doc = torch.zeros((n_chunks, 3, self.max_chunk_len), dtype=torch.long)
        for i in range(n_chunks):
            chunk_tokens = tokens[(self.max_chunk_len-2) * i : (self.max_chunk_len-2) * (i+1)]
            chunk_tokens.insert(0, "[CLS]")
            chunk_tokens.append("[SEP]")          
            chunk_tokens_ids = self.tokenizer.convert_tokens_to_ids(chunk_tokens)
                     
            attn_masks = [1] * len(chunk_tokens_ids)         
            # Pad the last chunk
            while len(chunk_tokens_ids) < self.max_chunk_len:
                chunk_tokens_ids.append(0)
                attn_masks.append(0)
                
            token_type_ids = [0] * self.max_chunk_len       
            assert len(chunk_tokens_ids) == self.max_chunk_len and len(attn_masks) == self.max_chunk_len
                         
            doc[i] = torch.cat((torch.LongTensor(chunk_tokens_ids).unsqueeze(0),
                                torch.LongTensor(attn_masks).unsqueeze(0),
                                torch.LongTensor(token_type_ids).unsqueeze(0)), dim=0)
        
        # Label tensor
        label = self.info_df.loc[idx, self.rob_item]
        label = torch.LongTensor([label])  
                
        return doc, label
    
    def cls_weight(self):
        df = self.info_df   
        n_pos = len(df[df[self.rob_item]==1])
        n_neg = len(df[df[self.rob_item]==0])     
        return [1/n_pos, 1/n_neg]
        

class PadDoc:
    def __call__(self, batch):
        # Element in batch: (doc, label)
        # Sort batch by num_chunks in descending order. x[0] => doc, x[1] -> label
        # x[0].shape = [num_chunks, 3, max_chunk_len]
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
#data_set = DocDataset(info_file = '/media/mynewdrive/rob/data/rob_info_a.pkl',
#                       pkl_dir = '/media/mynewdrive/rob/data/rob_str',
#                       rob_item = 'RandomizationTreatmentControl',
#                       max_chunk_len = 512, 
#                       cut_head_ratio = 0.1,
#                       cut_tail_ratio = 0.1,
#                       group = 'train')  # 'valid', 'test'        
#
#
#len(data_set)  # 6272
#data_set.cls_weight()
#idx = 0
#doc = data_set[idx][0]
#doc[0]   # 1st chunk
#doc[-1]  # last chunk
#
#
#n_chk = []
#for i in range(len(data_set)):
#    output = data_set[i][0]
#    n_chk.append(output.size()[0])
#    if i % 1000 == 0:
#        print(output.size())
#
## max num_chunk in 23 (train), 21 (valid), 25 (test)
#max(n_chk)  
#
#
## DataLoader
#from torch.utils.data import DataLoader
#data_loader = DataLoader(data_set, batch_size=16, shuffle=True, num_workers=4, collate_fn=PadDoc())
#
#
#batch = next(iter(data_loader))
#doc_batch = batch[0]; print(doc_batch.size())   
#label_batch = batch[1]; print(label_batch.size())    
#len_batch = batch[2]; print(len_batch)  
#
#doc_batch.size()  # [batch_size, num_chunks, 3, max_chunk_len]
#label_batch.size()  # [batch_size]
#len_batch.size()  # [batch_size]
#
#for i, batch in enumerate(data_loader):
#    if i % 50 == 0:
#        print("[batch {}] Doc: {}, Label: {}".format(i, batch[0].size(), batch[1].size()))

