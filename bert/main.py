#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:14:20 2020

@author: qwang
"""

import os
os.chdir('/home/qwang/rob/rob-kiwi/bert')

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from transformers import BertConfig, BertTokenizer, AdamW


from utils import metrics
from arg_parser import get_args
from data_loader import DocDataset, PadDoc

from model import BertLinear, BertLSTM
from train import train_evaluate


#%% Setting
# Get arguments from command line
args = get_args()

# random seed
random.seed(args.seed)
#np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False   # This makes things slower  

# device
if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    print('Use {} GPUs: '.format(torch.cuda.device_count()), device)
elif torch.cuda.device_count() == 1:
    device = torch.device("cuda")
    print('Use 1 GPU: ', device)
else:
    device = torch.device('cpu')     

        
#%% Create dataset and data loader  
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

train_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
                       rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='train', tokenizer=bert_tokenizer)


valid_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
                       rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='valid', tokenizer=bert_tokenizer)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())

#test_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
#                      rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
#                      cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
#                      group='test', tokenizer=bert_tokenizer)
#test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())

#%% Model
config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = args.num_labels
config.output_attentions = False
config.output_hidden_states = False
config.n_chunks = args.n_chunks
config.freeze_bert = args.freeze_bert
config.unfreeze_bert_last = args.unfreeze_bert_last

model = BertLinear(config)
#print(model)

# Demonstrate some pars
pars = list(model.named_parameters())
#print('\nBERT has {} named parameters.\n'.format(len(pars)))
#print('==== Embedding Layer ====\n')
#for p in pars[0:5]:
#    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#    
#print('\n==== First Transformer ====\n')
#for p in pars[5:21]:
#    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
#
#print('\n==== Output Layer ====\n')
#for p in pars[-4:]:
#    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    
n_pars = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
#print(model)
print("Number of parameters: {}".format(n_pars))

#%% Optimizer & Scheduler & Criterion
optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-8)
model = model.to(device)

# Learning rate
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * args.num_epochs
# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Weight balancing
if args.weight_balance == True and torch.cuda.device_count() == 0:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()))
elif args.weight_balance == True and torch.cuda.device_count() > 0:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()).cuda())
else:
    criterion = nn.CrossEntropyLoss()

if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)
criterion = criterion.to(device)    
    

#%% Train the model
train_evaluate(model, train_loader, valid_loader, optimizer, scheduler, criterion, metrics, args, device)


    
    
    
    
    