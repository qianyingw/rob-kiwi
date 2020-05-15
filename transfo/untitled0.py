#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:58:56 2020

@author: qwang
"""

from transformers import BertTokenizer, BertModel, BertForPreTraining, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_medium/')

model = BertLSTM.from_pretrained('/media/mynewdrive/rob/data/pre_wgts/bert_medium', config=config)

model = BertForPreTraining.from_pretrained('bert-base-uncased', from_tf=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', from_tf=True)
    

#%%
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)



# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for.Here is the sentence I want embeddings for."
new = text + text + text + text + text + text + text + text + text + text + text + text

marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(new)
len(tokenized_text)
# Print out the tokens.
print(tokenized_text)

