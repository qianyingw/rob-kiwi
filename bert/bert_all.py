#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:43:19 2020
@author: qwang

ref: https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, AdamW
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

os.chdir('/home/qwang/rob/rob-kiwi/bert')
from utils import metrics

#%% Setting
INFO_FILE = '/media/mynewdrive/rob/data/rob_info_a.pkl'
PKL_DIR = '/media/mynewdrive/rob/data/rob_str'
CUT_HEAD_RATIO = 0.1
CUT_TAIL_RATIO = 0.1
MAX_LEN = 500
ROB_ITEM = "RandomizationTreatmentControl"
BATCH_SIZE = 16
EPOCHS = 4
THRESHOLD = 0.5
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    print('Use {} GPUs: '.format(torch.cuda.device_count()), device)
elif torch.cuda.device_count() == 1:
    device = torch.device("cuda")
    print('Use 1 GPU: ', device)
else:
    device = torch.device('cpu')     
    


#%% Read token sent lists
### Add [CLS] and [SEP]; convert tokens to ids
def sent_list_to_token_ids(row):
    # Read tokens from each df row
    pkl_path = os.path.join(PKL_DIR, row['goldID']+'.pkl') 
    with open(pkl_path, 'rb') as fin:
        sent_ls = pickle.load(fin)    
        
    # Cut sentences from head/tail
    doc_len = len(sent_ls)
    n_head = int(doc_len * CUT_HEAD_RATIO)
    n_tail = int(doc_len * CUT_TAIL_RATIO)
    sent_ls = sent_ls[n_head : -n_tail]
    # Split sentence lists to tokens. Add [CLS] and [SEP] for BERT  
    tokens = []
    for l in sent_ls:
        tokens = tokens + l[0].split(" ")
        tokens.append("[SEP]")
    tokens.insert(0, "[CLS]")
    ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    return ids

df = pd.read_pickle(INFO_FILE)  # From rob_kiwi/data_helper.py

# Split info data
train_df = df[df.partition == "train"]
valid_df = df[df.partition == "valid"] 
test_df = df[df.partition == "test"] 

# Obtain doc token ids 
train_tokenId = train_df.apply(sent_list_to_token_ids, axis=1)   
valid_tokenId = valid_df.apply(sent_list_to_token_ids, axis=1)     
test_tokenId = test_df.apply(sent_list_to_token_ids, axis=1) 

# Obtain labels
train_label = train_df[ROB_ITEM].values
valid_label = valid_df[ROB_ITEM].values
test_label = test_df[ROB_ITEM].values

#%% About token_en
token_len = [len(doc) for doc in train_tokenId]

import matplotlib.pyplot as plt
n_bins = 20
plt.hist(token_len, n_bins, alpha=0.8, ec='black')
plt.xlabel('token length')
plt.ylabel('count')
plt.show()

print('Max token length: ', max(token_len))
print('# docs > 6k tokens: ', len([t for t in token_len if t > 6000]))
print('# docs > 7k tokens: ', len([t for t in token_len if t > 7000]))
print('# docs > 8k tokens: ', len([t for t in token_len if t > 8000]))
print('# docs > 10k tokens: ', len([t for t in token_len if t > 10000]))

#%% Pad & Truncate
from keras.preprocessing.sequence import pad_sequences

print('Pad/truncate sents to {}'.format(MAX_LEN))
print('Pad token: "{:}", ID: {:}'.format(bert_tokenizer.pad_token, bert_tokenizer.pad_token_id))

# Pad input tokens with value 0 (id for [PAD])
train_tokenId = pad_sequences(train_tokenId, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
valid_tokenId = pad_sequences(valid_tokenId, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

#%% Create attention masks
# For each doc: if token ID is 0 , set mask to 0 (pad); else set mask to 1 (real token)
train_mask = []
for doc in train_tokenId:  
    mask = [int(token_id > 0) for token_id in doc]
    train_mask.append(mask)
    
valid_mask = []
for doc in valid_tokenId:  
    mask = [int(token_id > 0) for token_id in doc]
    valid_mask.append(mask)
    
# Convert to torch tensors
train_tokenId = torch.tensor(train_tokenId)
valid_tokenId = torch.tensor(valid_tokenId)

train_label = torch.tensor(train_label)
valid_label = torch.tensor(valid_label)

train_mask = torch.tensor(train_mask)
valid_mask = torch.tensor(valid_mask)

#%% Creat data loader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Dataloader for train
train_data = TensorDataset(train_tokenId, train_mask, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set.
valid_data = TensorDataset(valid_tokenId, valid_mask, valid_label)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE)

#%% Model
#model = BertForSequenceClassification.from_pretrained(
#    "bert-base-uncased",
#    num_labels = 2,
#    output_attentions = False, 
#    output_hidden_states = False
#)
#
#model.cuda()

config = BertConfig.from_pretrained('bert-base-uncased')
config.num_labels = 2
config.output_attentions = False
config.output_hidden_states = False

model = BertForSequenceClassification(config)
print(model)

# Demonstrate some pars
pars = list(model.named_parameters())
print('\nBERT has {} named parameters.\n'.format(len(pars)))
print('==== Embedding Layer ====\n')
for p in pars[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in pars[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')
for p in pars[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    

# Optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
model = model.to(device)


# Learning rate
from transformers import get_linear_schedule_with_warmup
total_steps = len(train_dataloader) * EPOCHS
# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)





#%% Train
def train(model, data_loader, optimizer, scheduler, metrics, threshold=0.5):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    
    model.train()
    
    with tqdm(total=len_iter) as progress_bar:
        for batch in data_loader:
            
            tokens_id, attn_mask, labels = batch    
            model.zero_grad() 
            
            output = model(tokens_id, 
                           token_type_ids = None, 
                           attention_mask = attn_mask, 
                           labels = labels)
            loss, logits = output[:2]        
            preds = nn.Softmax(dim=1)(logits)
            
            epoch_scores = metrics(preds, labels, threshold)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
            optimizer.step()
            scheduler.step()
                    
            scores['loss'] += loss.item()
            for key, value in epoch_scores.items():               
                scores[key] += value        
            progress_bar.update(1)  # update progress bar 
    
    for key, value in scores.items():
        scores[key] = value / len_iter   
    return scores

#%% Evaluate   
def evaluate(model, data_loader, optimizer, scheduler, metrics, threshold=0.5):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in data_loader:
                
                tokens_id, attn_mask, labels = batch          
                output = model(tokens_id, 
                               token_type_ids = None, 
                               attention_mask = attn_mask,
                               labels = labels)
                
                loss, logits = output[:2]
                preds = nn.Softmax(dim=1)(logits)
                
                epoch_scores = metrics(preds, labels, threshold)
                
                scores['loss'] += loss.item()
            for key, value in epoch_scores.items():               
                scores[key] += value        
            progress_bar.update(1)  # update progress bar        
    return scores

#%% train_eval
for epoch in range(EPOCHS):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(train_dataloader)
    
    # Create args and output dictionary (for json output)
    output_dict = {'prfs': {}}
 
    train_scores = train(model, train_dataloader, optimizer, scheduler, metrics, THRESHOLD)
    valid_scores = evaluate(model, valid_dataloader, optimizer, scheduler, metrics, THRESHOLD) 
    
    output_dict['prfs'][str('train_'+str(epoch+1))] = scores

    # Update output dictionary
    output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
    output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
    
    print("\n\nEpoch {}/{}...".format(epoch+1, EPOCHS))                       
    print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%'.format(
        train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
    print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%\n'.format(
        valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))