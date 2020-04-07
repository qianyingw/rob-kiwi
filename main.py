#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:24:36 2019

@author: qwang
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

os.chdir('/home/qwang/rob/rob-kiwi')


from utils import metrics
from arg_parser import get_args
from data_loader import RoBDataset, PadDoc
from model import ConvNet
from train import train_evaluate


#%% Get arguments from command line
args = get_args()
#args_dict = vars(args)

## Save args to json
#with open(os.path.join(log_dir, 'args.json'), 'w') as fout:
#    json.dump(args_dict, fout, indent=4)

#%% Set seed and device
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


#%% Set logger
#log_dir = os.path.join(args.exp_path, args.exp_name)
#if os.path.exists(log_dir) == False:
#    os.makedirs(log_dir)       
##utils.set_logger(os.path.join(log_dir, 'train.log'))


        
#%% Create dataset and loader
train_set = RoBDataset(info_file = args.info_path, mat_dir = args.pkl_dir, 
                       rob_item = args.rob_item, max_len = args.max_doc_len,
                       group = 'train')

valid_set = RoBDataset(info_file = args.info_path, mat_dir = args.pkl_dir, 
                       rob_item = args.rob_item, max_len = args.max_doc_len,
                       group = 'valid')

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())


#test_set = RoBDataset(info_file = args.info_path, mat_dir = args.pkl_dir, 
#                       rob_item = args.rob_item, max_len = args.max_doc_len, 
#                       group='test')
#test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=PadDoc())

#%% Define the model
if args.net_type == 'cnn':
    sizes = args.filter_sizes.split(',')
    sizes = [int(s) for s in sizes]
    model = ConvNet(tune_embed = args.tune_embed,
                    input_len = args.max_doc_len,
                    embedding_dim = args.embed_dim, 
                    n_filters = args.num_filters, 
                    filter_sizes = sizes, 
                    output_dim = 2, 
                    dropout = args.dropout)

n_pars = sum(p.numel() for p in model.parameters())
print(model)
print("Number of parameters: {}".format(n_pars))


#%% Define the optimizer, criterion and metrics
optimizer = optim.Adam(model.parameters())
metrics_fn = metrics

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
train_evaluate(model, train_loader, valid_loader, criterion, optimizer, metrics_fn, args)
