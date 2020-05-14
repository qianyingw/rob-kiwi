#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#    Run this script once to get a PyTorch model. 
#    Then put pytorch_model.bin, bert_config.json and vocab.txt in a folder and compress
#    
#    convert_tf_checkpoint_to_pytorch
#        https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_original_tf_checkpoint_to_pytorch.py  
#            
#    BioBERT in pytorch
#        When you use its checkpoint for pytorch, optimizer's parameters need to be excluded (transformers.load_tf_weights_in_bert has done this)
#        https://github.com/dmis-lab/biobert/issues/2#issuecomment-458805972
#    
#    transformers.load_tf_weights_in_bert 
#        https://github.com/huggingface/transformers/blob/94cb73c2d2efeb188b522ff352f98b15124ba9f8/src/transformers/modeling_bert.py#L61
#        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v which are not required for using pretrained model
#
#       
#    Created on Wed May  14 12:58:56 2020
#    @author: qwang
"""

import os
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from transformers import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert



def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    
    config = BertConfig.from_json_file(bert_config_file)  # Initialise PyTorch model
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    load_tf_weights_in_bert(model, config, tf_checkpoint_path)  # Load weights from tf checkpoint

    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)  # Save pytorch-model
    
    
    
def convert_tf_checkpoint_to_pytorch_albert(tf_checkpoint_path, albert_config_file, pytorch_dump_path):
    
    config = AlbertConfig.from_json_file(albert_config_file)  # Initialise PyTorch model
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = AlbertForPreTraining(config)

    load_tf_weights_in_albert(model, config, tf_checkpoint_path)  # Load weights from tf checkpoint

    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)  # Save pytorch-model

#%% BERT-Medium
# Downloaded from https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip
wgt_dir = "/media/mynewdrive/rob/data/pre_wgts/source/uncased_L-8_H-512_A-8"
bert_config_file = os.path.join(wgt_dir, "bert_config.json")
tf_checkpoint_path = os.path.join(wgt_dir, "bert_model.ckpt")
pytorch_dump_path = os.path.join(wgt_dir, "pytorch_model.bin")

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)

#%% ALBERT-base  
# Download from https://storage.googleapis.com/albert_models/albert_base_v2.tar.gz
wgt_dir = "/media/mynewdrive/rob/data/pre_wgts/source/albert_base_v2"
albert_config_file = os.path.join(wgt_dir, "albert_config.json")
tf_checkpoint_path = os.path.join(wgt_dir, "model.ckpt-best")
pytorch_dump_path = os.path.join(wgt_dir, "pytorch_model.bin")

convert_tf_checkpoint_to_pytorch_albert(tf_checkpoint_path, albert_config_file, pytorch_dump_path)

#%% BioBERT    
# Download from https://github.com/dmis-lab/biobert
wgt_dir = "/media/mynewdrive/rob/data/pre_wgts/source/biobert_v1.1_pubmed"

bert_config_file = os.path.join(wgt_dir, "bert_config.json")
tf_checkpoint_path = os.path.join(wgt_dir, "model.ckpt-1000000")
pytorch_dump_path = os.path.join(wgt_dir, "pytorch_model.bin")

convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)