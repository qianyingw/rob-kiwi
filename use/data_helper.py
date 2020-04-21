#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:10:07 2020

@author: qwang
"""
import json
import os
import pandas as pd
import math
from tqdm import tqdm
import random
import pickle


# Change to data dir
data_dir = '/media/mynewdrive/rob/'
os.chdir(data_dir)

seed = 1234
train_ratio = 0.8
val_ratio = 0.1
data_json_path = 'data/rob_tokens.json'
rm_na = False


def read_json(json_path):
    df = []
    with open(json_path, 'r') as fin:
        for line in fin:
            df.append(json.loads(line))
    return df


#%% Generate string list pkl files
def gen_ls_pkls(dat_ls, pkl_dir): 
    
    if os.path.exists(pkl_dir) == False:
        os.makedirs(pkl_dir) 
        
    for i, r in tqdm(enumerate(dat_ls)):  
        doc_ls = []             
        for s in r['sentTokens']:
            sent_str = [' '.join(s)]
            doc_ls.append(sent_str)    
            
        pkl_path = os.path.join(pkl_dir, r['goldID']+'.pkl')        
        with open(pkl_path, 'wb') as fout:
            pickle.dump(doc_ls, fout)


dat = read_json(data_json_path)
gen_ls_pkls(dat_ls=dat, pkl_dir='data/rob_str')   


#with open(pkl_path, 'rb') as fin:
#    ls = pickle.load(fin)

#%% Generate document matrix (dataframe) pkls
# USE
import tensorflow_hub as hub
embed_func = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def gen_df_pkls(dat_ls, pkl_dir): 
    
    if os.path.exists(pkl_dir) == False:
        os.makedirs(pkl_dir) 
        
    for i, r in tqdm(enumerate(dat_ls)):                
        d_mat = []             
        for sl in r['sentTokens']:
            s = [' '.join(sl)]
            s_vec = embed_func(s).numpy().astype('float16') # convert sentence to embedding 
            s_vec = s_vec.tolist()[0]
            d_mat.append(s_vec)        
        mat_df = pd.DataFrame(d_mat)
        pkl_path = os.path.join(pkl_dir, r['goldID']+'.pkl')
        mat_df.to_pickle(pkl_path)

dat = read_json(data_json_path)
gen_df_pkls(dat_ls=dat, pkl_dir='data/rob_mat')   


#%% rob_info_a(b).pkl
dat = read_json(data_json_path)

if rm_na == True:  # Remove missing records for conceal/welfare/conflict
    dat = [g for g in dat if math.isnan(g['AllocationConcealment']) == False]  
    
# Shuffle data
random.seed(seed)
random.shuffle(dat)
train_size = int(len(dat) * train_ratio)
val_size = int(len(dat) * val_ratio)

# Add 'partition' for random/blind/size/exclusion
for g in dat[:train_size]:
    g['partition'] = 'train'
for g in dat[train_size : (train_size + val_size)]:
    g['partition'] = 'valid'
for g in dat[(train_size + val_size):]:
    g['partition'] = 'test'    

for i, r in tqdm(enumerate(dat)):          
    del r['wordTokens']; del r['sentTokens']
    del r['fileLink']; del r['DocumentLink']; del r['txtLink']

df = pd.DataFrame(dat)

if rm_na == False:
    df.to_pickle('data/rob_info_a.pkl')  # 'rob_info_a.pkl' for random/blind/size/exclusion
else:
    df.to_pickle('data/rob_info_b.pkl')  # 'rob_info_b.pkl' for conceal/welfare/conflict


#info = pd.read_pickle('data/rob_info_a.pkl')
#for i in range(len(info)):
#    info.loc[i, 'group'] = re.sub(r'\d+', "", info.loc[i, 'goldID'])
#df = info[info.group == 'iicarus']
    
## Generate single pkl
#def gen_pkl(dat_ls, pkl_path):
#    for i, r in tqdm(enumerate(dat_ls)): 
#        sentTokens = r['sentTokens']  
#        
#        del r['wordTokens']; del r['sentTokens']
#        del r['fileLink']; del r['DocumentLink']; del r['txtLink']
#        
#        d_mat = []             
#        for sl in sentTokens:
#            s = [' '.join(sl)]
#            s_vec = embed_func(s).numpy().astype('float16') # convert sentence to embedding 
#            s_vec = s_vec.tolist()[0]
#            d_mat.append(s_vec)
#        dat_ls[i]['docMat'] = d_mat
#    df = pd.DataFrame(dat_ls)
#    df.to_pickle(pkl_path)

