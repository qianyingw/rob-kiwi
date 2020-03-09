#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:40:29 2019
github.com/CSTR-Edinburgh/mlpractical/blob/mlp2019-20/mlp_cluster_tutorial/arg_extractor.py
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
    parser.add_argument('--batch_size', nargs="?", type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=20, help='Number of epochs')   
    
    parser.add_argument('--args_json_path', nargs="?", type=str, default=None, help='Path of argument json file')
    parser.add_argument('--exp_dir', nargs="?", type=str, default="/media/qwang/rob/temp2", help='Folder of the experiment')
    
    parser.add_argument('--save_model', nargs="?", type=str2bool, default=False, help='Save model.pth.tar with best loss')
    parser.add_argument('--stop_patience', nargs="?", type=int, default=5, help='Number of cases when valid loss is lower than the best loss')
    parser.add_argument('--stop_criterion', nargs="?", type=float, default=0.02, help='Acceptable difference compared with the best loss')

    
    # Data
    parser.add_argument('--info_path', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_info_a.pkl", help='Path of pkl info file')
    parser.add_argument('--pkl_dir', nargs="?", type=str, default="/media/mynewdrive/rob/data/rob_mat", help='Dir of pkl files')   
    # RoB item
    parser.add_argument('--rob_item', nargs="?", type=str, default="RandomizationTreatmentControl", 
                        choices=['RandomizationTreatmentControl'
                                 'BlindedOutcomeAssessment'
                                 'SampleSizeCalculation'
                                 'AnimalExclusions'     
                                 'AllocationConcealment'
                                 'AnimalWelfareRegulations'
                                 'ConflictsOfInterest'], 
                        help='Risk of bias item')
    
    # Model
    parser.add_argument('--net_type', nargs="?", type=str, default='cnn', 
                        choices=['cnn', 'rnn', 'attn', 'transformer'], 
                        help="Different networks [options: 'cnn', 'rnn', 'attn', 'transformer']")
    parser.add_argument('--dropout', nargs="?", type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--embed_dim', nargs="?", type=int, default=512, help='Dimension of sentence encoder')   
    parser.add_argument('--weight_balance', nargs="?", type=str2bool, default=False, help='Assign class weights for imbalanced data')
    parser.add_argument('--tune_embed', nargs="?", type=str2bool, default=True, help='Tune embedding or not')
    parser.add_argument('--max_doc_len', nargs="?", type=int, default=400, help='Maximum number of sents in one document overall the batches')
    
    # CNN
    parser.add_argument('--num_filters', nargs="?", type=int, default=100, help='Number of filters for each filter size (cnn)')   
    parser.add_argument('--filter_sizes', nargs="?", type=str, default='3,4,5', help='Filter sizes (cnn)')
    
    # RNN/Attention
    parser.add_argument('--rnn_cell_type', nargs="?", type=str, default="lstm", choices=['lstm', 'gru'], help="Type of RNN cell [options: 'lstm', 'gru']")
    parser.add_argument('--rnn_hidden_dim', nargs="?", type=int, default=100, help='Number of features in RNN hidden state')
    parser.add_argument('--rnn_num_layers', nargs="?", type=int, default=1, help='Number of recurrent layers')
    parser.add_argument('--bidirection', nargs="?", type=str2bool, default=False, help='Apply the bidirectional RNN')

    
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

    with open(json_file_path) as fin:
        args_dict = json.load(fp=fin)

    for key, value in vars(existing_args_dict).items():
        if key not in args_dict:
            args_dict[key] = value

    args_dict = AttributeAccessibleDict(args_dict)
    return args_dict
