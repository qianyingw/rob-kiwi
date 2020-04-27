#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:14:20 2020

@author: qwang
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

os.chdir('/home/qwang/rob/rob-kiwi/bert')


from utils import metrics
from arg_parser import get_args
from data_loader import DocDataset, PadDoc


#from model import ConvNet, AttnNet
#from train import train_evaluate


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
train_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
                       rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='train')

valid_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
                       rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
                       cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
                       group='valid')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())


#test_set = DocDataset(info_file=args.info_file, pkl_dir=args.pkl_dir, 
#                      rob_item=args.rob_item, max_chunk_len=args.max_chunk_len,
#                      cut_head_ratio=args.cut_head_ratio, cut_tail_ratio=args.cut_tail_ratio,
#                      group='test')
#test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())


#%% Model