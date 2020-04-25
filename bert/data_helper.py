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
from torch.utils.data import Dataset

from transformers import BertTokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#%% Setting
args = {"info_file": "/media/mynewdrive/rob/data/rob_info_a.pkl",
        "pkl_dir": "/media/mynewdrive/rob/data/rob_str",
        "cut_head_ratio": 0.1,
        "cut_tail_ratio": 0.1,
        "max_chunk_len": 512
        
 
        }

    
#%%
class DocDataset(Dataset):
    """
    info_file: 
        'rob_info_a.pkl' (for RandomizationTreatmentControl | BlindedOutcomeAssessment | SampleSizeCalculation | AnimalExclusions)     
        'rob_info_b.pkl' (for AllocationConcealment | AnimalWelfareRegulations | ConflictsOfInterest)
    
    group: 'train', 'valid', 'test'   
    output: [num_chunks, 3, max_chunk_len]
            '3' refers to tokens_ids, token_type_ids, attn_masks
        
    """
    def __init__(self, info_file, pkl_dir, rob_item, max_chunk_len, cut_head_ratio, cut_tail_ratio, group):  
        
        self.pkl_dir = pkl_dir
        self.rob_item = rob_item   
        self.max_chunk_len = max_chunk_len
        self.cut_head_ratio = cut_head_ratio
        self.cut_tail_ratio = cut_tail_ratio
        
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
            
#        assert n_chunks <= 20, "The document is too large. Try to increase cut_head/tail_ratio."
            
        output = torch.zeros(size=(n_chunks, 3, self.max_chunk_len), dtype=torch.long)
        for i in range(n_chunks):
            chunk_tokens = tokens[(self.max_chunk_len-2) * i : (self.max_chunk_len-2) * (i+1)]
            chunk_tokens.insert(0, "[CLS]")
            chunk_tokens.append("[SEP]")          
            chunk_tokens_ids = bert_tokenizer.convert_tokens_to_ids(chunk_tokens)
                     
            attn_masks = [1] * len(chunk_tokens_ids)         
            # Pad the last chunk
            while len(chunk_tokens_ids) < self.max_chunk_len:
                chunk_tokens_ids.append(0)
                attn_masks.append(0)
                
            token_type_ids = [0] * self.max_chunk_len       
            assert len(chunk_tokens_ids) == self.max_chunk_len and len(attn_masks) == self.max_chunk_len
                 
            output[i] = torch.cat((torch.LongTensor(chunk_tokens_ids).unsqueeze(0),
                                   torch.LongTensor(token_type_ids).unsqueeze(0),
                                   torch.LongTensor(attn_masks).unsqueeze(0)), dim=0)
                
        return output
        
        
#%% Instance   
train_set = DocDataset(info_file = '/media/mynewdrive/rob/data/rob_info_a.pkl',
                       pkl_dir = '/media/mynewdrive/rob/data/rob_str',
                       rob_item = 'RandomizationTreatmentControl',
                       max_chunk_len = 512, 
                       cut_head_ratio = 0.1,
                       cut_tail_ratio = 0.1,
                       group = 'train')        

valid_set = DocDataset(info_file = '/media/mynewdrive/rob/data/rob_info_a.pkl',
                       pkl_dir = '/media/mynewdrive/rob/data/rob_str',
                       rob_item = 'RandomizationTreatmentControl',
                       max_chunk_len = 512, 
                       cut_head_ratio = 0.1,
                       cut_tail_ratio = 0.1,
                       group = 'valid') 

test_set = DocDataset(info_file = '/media/mynewdrive/rob/data/rob_info_a.pkl',
                       pkl_dir = '/media/mynewdrive/rob/data/rob_str',
                       rob_item = 'RandomizationTreatmentControl',
                       max_chunk_len = 512, 
                       cut_head_ratio = 0.1,
                       cut_tail_ratio = 0.1,
                       group = 'test') 

len(train_set)  # 6272
#train_set.cls_weight()

n_chk = []
for i in range(len(test_set)):
    output = test_set[i]
    n_chk.append(output.size()[0])
    if i % 1000 == 0:
        print(output.size())

# max num_chunk in 23 (train), 21 (valid), 25 (test)
max(n_chk)  


