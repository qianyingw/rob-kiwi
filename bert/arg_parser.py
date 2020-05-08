#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 11:25:41 2020

@author: qwang
"""

import argparse
import json
import os

USER = os.getenv('USER')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(description='RoB training and inference helper script')

    
    # Experiments
    parser.add_argument('--seed', nargs="?", type=int, default=1234, help='Seed for random number generator')
    parser.add_argument('--batch_size', nargs="?", type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=2, help='Number of epochs')    
    parser.add_argument('--train_ratio', nargs="?", type=float, default=0.8, help='Ratio of training set')
    parser.add_argument('--val_ratio', nargs="?", type=float, default=0.1, help='Ratio of validation set')
    parser.add_argument('--lr', nargs="?", type=float, default=2e-5, help='AdamW learning rate')
    parser.add_argument('--clip', nargs="?", type=float, default=0.1, help='Gradient clipping')
    parser.add_argument('--accum_step', nargs="?", type=int, default=4, help='Number of steps for gradient accumulation')
    
    parser.add_argument('--info_file', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_info_a.pkl", help='Path of info pickle file')
    parser.add_argument('--pkl_dir', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_str", help='Directory of pickle files')
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file')
    parser.add_argument('--exp_dir', nargs="?", type=str, default="/home/qwang/rob/bert", help='Folder of the experiment')
    parser.add_argument('--save_model', nargs="?", type=str2bool, default=False, help='Save model.pth.tar with best loss')
    parser.add_argument('--fp16', nargs="?", type=str2bool, default=False, help='Train with half precision')
       
    # RoB item
    parser.add_argument('--rob_item', nargs="?", type=str, default="RandomizationTreatmentControl", 
                        choices=['RandomizationTreatmentControl',
                                 'BlindedOutcomeAssessment',
                                 'SampleSizeCalculation',
                                 'AnimalExclusions',
                                 'AllocationConcealment',
                                 'AnimalWelfareRegulations',
                                 'ConflictsOfInterest'], 
                        help='Risk of bias item')

    # Model
    parser.add_argument('--net_type', nargs="?", type=str, default='bert_lstm', 
                        choices=['bert_linmax', 'bert_linavg', 'bert_lstm',
                                 'albert_linmax', 'albert_linavg', 'albert_lstm'], 
                        help="Different network models")
    parser.add_argument('--freeze_model', nargs="?", type=str, default='bert_encoder', 
                        choices=['bert', 'bert_encoder', 'albert', 'albert_encoder'], 
                        help='Options of freezing bert/albert parameters')
    
    parser.add_argument('--weight_balance', nargs="?", type=str2bool, default=True, help='Assign class weights for imbalanced data')
    parser.add_argument('--threshold', nargs="?", type=float, default=0.5, help='Threshold for positive class value')
    
    parser.add_argument('--max_chunk_len', nargs="?", type=int, default=512, help='Max context window size for bert')
    parser.add_argument('--cut_head_ratio', nargs="?", type=float, default=0.1, help='Ratio of tokens cut from head')
    parser.add_argument('--cut_tail_ratio', nargs="?", type=float, default=0.1, help='Ratio of tokens cut from tail')
    
    # BERT/ALBERT
    parser.add_argument('--num_labels', nargs="?", type=int, default=2, help='Number of output labels')
    parser.add_argument('--max_n_chunk', nargs="?", type=int, default=20, help='Max number of text chunks')
    parser.add_argument('--num_hidden_layers', nargs="?", type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', nargs="?", type=int, default=2, help='Number of attention heads')
    parser.add_argument('--hidden_size', nargs="?", type=int, default=768, help='Number of hidden units')
    
   
    args = parser.parse_args()
    
    if args.args_json_path is None:
        arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
        print(arg_str)
    else:
        args = extract_args_from_json(json_file_path=args.args_json_path, existing_args_dict=args)   
    
    return args


class AttributeAccessibleDict(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def extract_args_from_json(json_file_path, existing_args_dict=None):

    summary_filename = json_file_path
    with open(summary_filename) as fin:
        args_dict = json.load(fp=fin)

    for key, value in vars(existing_args_dict).items():
        if key not in args_dict:
            args_dict[key] = value

    args_dict = AttributeAccessibleDict(args_dict)

    return args_dict
