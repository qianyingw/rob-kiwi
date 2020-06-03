#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:14:20 2020

@author: qwang
"""

import os
#os.chdir('/home/qwang/rob/rob-kiwi/transfo')

import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import BertConfig, BertTokenizer, AdamW
from transformers import AlbertConfig, AlbertTokenizer
from transformers import XLNetConfig, XLNetTokenizer

from utils import metrics
from arg_parser import get_args
from data_loader import DocDataset, PadDoc

from model import BertPoolLSTM, BertLSTM, AlbertLinear, AlbertLSTM
from model_xlnet import XLNetLinear, XLNetLSTM, XLNetConv
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


#%% Tokenizer & Config & Model
if args.net_type == "bert_pool_lstm":
    # Default: rob/data/pre_wgts/bert_medium/
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.wgts_dir, do_lower_case=True)  
    # Config
    config = BertConfig.from_pretrained(args.wgts_dir)  
    config.output_hidden_states = True
    config.num_labels = args.num_labels
    config.unfreeze = args.unfreeze
    config.pool_method = args.pool_method
    config.pool_layers = args.pool_layers
      
    if args.num_hidden_layers:  config.num_hidden_layers = args.num_hidden_layers
    if args.num_attention_heads:  config.num_attention_heads = args.num_attention_heads
    
    model = BertPoolLSTM.from_pretrained(args.wgts_dir, config=config)

    


# if args.net_type in ["bert_linear", "bert_lstm"]:
#     # Default: rob/data/pre_wgts/bert_medium/
#     # Tokenizer
#     tokenizer = BertTokenizer.from_pretrained(args.wgts_dir, do_lower_case=True)  
#     # Config
#     config = BertConfig.from_pretrained(args.wgts_dir)  
#     config.summary_type = 'first'  # for SequenceSummary
#     config.num_labels = args.num_labels
#     config.unfreeze = args.unfreeze
      
    
#     if args.num_hidden_layers:  config.num_hidden_layers = args.num_hidden_layers
#     if args.num_attention_heads:  config.num_attention_heads = args.num_attention_heads
    
#     # Model
#     if args.net_type == "bert_lstm":
#         model = BertLSTM.from_pretrained(args.wgts_dir, config=config)
#     else:
#         model = BertLinear.from_pretrained(args.wgts_dir, config=config)
    
    
# if args.net_type in ["xlnet_linear", "xlnet_lstm", "xlnet_conv"]:
    
#     # Tokenizer
#     tokenizer = XLNetTokenizer.from_pretrained(args.wgts_dir, do_lower_case=True)  
#     # Config
#     config = XLNetConfig.from_pretrained(args.wgts_dir)    
#     config.num_labels = args.num_labels
#     # config.unfreeze = args.unfreeze   
#     config.n_layer = args.num_hidden_layers if args.num_hidden_layers else 12
#     config.n_head = args.num_attention_heads if args.num_attention_heads else 12
#     config.d_model = args.hidden_size if args.hidden_size else 768

#     # Model
#     if args.net_type == "xlnet_linear":
#         model = XLNetLinear.from_pretrained(args.wgts_dir, config=config)      
#     elif args.net_type == "xlnet_lstm":
#         model = XLNetLSTM.from_pretrained(args.wgts_dir, config=config)
#     else: # args.net_type == "xlnet_conv"
#         sizes = args.filter_sizes.split(',')
#         config.filter_sizes = [int(s) for s in sizes]
#         config.n_filters = args.num_filters
#         model = XLNetConv.from_pretrained(args.wgts_dir, config=config)
        
    

#if args.net_type.split('_')[0] == "bert":
#    tokenizer = BertTokenizer.from_pretrained(args.wgts_dir, do_lower_case=True)  # default: bert_medium
#    config = BertConfig.from_pretrained(args.wgts_dir)  
#    model = BertLSTM.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_medium/', config=config)
#
#if args.net_type.split('_')[0] == "albert": 
#    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2', do_lower_case=True)
#    config = AlbertConfig.from_pretrained('albert-base-v2')  # albert-large-v2'

# Common configs
#if args.hidden_size: 
#    config.hidden_size = args.hidden_size

# Demonstrate some pars
#print(model)

n_pars = sum(p.numel() for p in model.parameters())
print("\n========== All parameters: {} ===============================".format(n_pars))
for p in model.named_parameters():
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print("=====================================================================\n")

n_pars = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
print("========== Trainable parameters: {} ===========================".format(n_pars))
for p in model.named_parameters():
    if p[1].requires_grad == True:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))        
print("====================================================================\n")


#%% Create dataset and data loader  
train_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, rob_item=args.rob_item, 
                       max_chunk_len=args.max_chunk_len, max_n_chunk=args.max_n_chunk,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='train', tokenizer=tokenizer)
#temp = train_set[0][0]
valid_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, rob_item=args.rob_item, 
                       max_chunk_len=args.max_chunk_len, max_n_chunk=args.max_n_chunk,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='valid', tokenizer=tokenizer)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())

#test_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, rob_item=args.rob_item,  
#                      max_chunk_len=args.max_chunk_len, max_n_chunk=args.max_n_chunk,
#                      cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
#                      group='test', tokenizer=bert_tokenizer)
#test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
        
#%% 
# Optimizer
optimizer = AdamW(model.parameters(), lr = args.lr, eps = 1e-8)

# Slanted triangular Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_loader) * args.num_epochs // args.accum_step
warm_steps = int(total_steps * args.warm_frac)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=total_steps)

# Criterion (weight balancing)
if args.weight_balance == True and torch.cuda.device_count() == 0:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()))
elif args.weight_balance == True and torch.cuda.device_count() > 0:
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(train_set.cls_weight()).cuda())
else:
    criterion = nn.CrossEntropyLoss()

# Sent to device
if torch.cuda.device_count() > 1:  # multiple GPUs
    model = nn.DataParallel(module=model)
model = model.to(device)
criterion = criterion.to(device)    


#%% Train the model
train_evaluate(model, train_loader, valid_loader, optimizer, scheduler, criterion, metrics, args, device)


    
    
    
    
    