#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:10:07 2020

@author: qwang
"""
import json
import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import random

# USE
import tensorflow_hub as hub
embed_func = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Change to data dir
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

seed = 1234
train_ratio = 0.8
val_ratio = 0.1
data_json_path = 'data/rob_tokens.json'
rm_na = False

#%%
def read_json(json_path):
    df = []
    with open(json_path, 'r') as fin:
        for line in fin:
            df.append(json.loads(line))
    return df

# Generate pkl file
def gen_pkl(dat_ls, pkl_path):
    for i, r in tqdm(enumerate(dat_ls)): 
        sentTokens = r['sentTokens'] 
        d_mat = []        
        del r['wordTokens']; del r['sentTokens']
        del r['fileLink']; del r['DocumentLink']; del r['txtLink']
                     
        for sl in sentTokens:
            s = [' '.join(sl)]
            s_vec = embed_func(s).numpy().astype('float16') 
            s_vec = s_vec.tolist()[0]
            d_mat.append(s_vec)
        dat_ls[i]['docMat'] = d_mat
    df = pd.DataFrame(dat_ls)
    df.to_pickle(pkl_path)
    
#%%
dat = read_json(data_json_path)

# Remove na records for welfare/conflict/conceal
if rm_na == True:
    dat = [g for g in dat if math.isnan(g['welfare']) == False] 

# Shuffle data
random.seed(seed)
random.shuffle(dat)

train_size = int(len(dat) * train_ratio)
val_size = int(len(dat) * val_ratio)

train_ls = dat[:train_size]
val_ls = dat[train_size : (train_size + val_size)]
test_ls = dat[(train_size + val_size):]

gen_pkl(dat_ls=train_ls, pkl_path="data/train.pkl")
gen_pkl(dat_ls=val_ls, pkl_path="data/val.pkl")
gen_pkl(dat_ls=test_ls, pkl_path="data/test.pkl")

        


   
