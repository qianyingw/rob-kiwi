#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:28:44 2019
github.com/CSTR-Edinburgh/mlpractical/blob/mlp2019-20/mlp_cluster_tutorial/experiment_builder.py
@author: qwang

"""

import os
import logging
import json
from tqdm import tqdm

import torch
import utils

     

def train(model, iterator, criterion, optimizer, metrics):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(iterator)
    
    model.train()
    
    with tqdm(total=len_iter) as progress_bar:
        
        for batch in iterator:
            
            batch_doc = batch[0]   # [batch_size, doc_len, embed_dim]
            batch_label = batch[1]
            # batch_len = batch[2]
            
            optimizer.zero_grad()
            
            preds = model(batch_doc)  # preds.shape: [batch_size, output_dim]
            
            loss = criterion(preds, batch_label)       
            epoch_scores = metrics(preds, batch_label)  # dictionary of 5 metric scores
                   
            loss.backward()
            optimizer.step()
            
            scores['loss'] += loss.item()
            for key, value in epoch_scores.items():               
                scores[key] += value        
            progress_bar.update(1)  # update progress bar 
    
    for key, value in scores.items():
        scores[key] = value / len_iter
    
    return scores


def evaluate(model, iterator, criterion, metrics):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(iterator)
    
    model.eval()
    
    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in iterator:  
                
                batch_doc = batch[0]
                batch_label = batch[1]
                # batch_len = batch[2]
                
                preds = model(batch_doc)
                
                loss = criterion(preds, batch_label)
                epoch_scores = metrics(preds, batch_label)  # dictionary of 5 metric scores
                
                scores['loss'] += loss.item()
                for key, value in epoch_scores.items():               
                    scores[key] += value                        
                progress_bar.update(1)
                        
        for key, value in scores.items():
            scores[key] = value / len_iter
    
    return scores



def train_evaluate(model, train_iterator, valid_iterator, criterion, optimizer, metrics, args, restore_file=None):
    """
    
    """
    
    if os.path.exists(args.exp_dir) == False:
        os.makedirs(args.exp_dir)     
    
    if restore_file is not None:
        restore_path = os.path.join(args.exp_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}...".format(restore_path))  
        utils.load_checkpoint(restore_path, model, optimizer)
          
    best_valid_f1 = -float('inf')
    best_valid_loss = float('inf')
    
    # Create args and output dict
    output_dict = {'args': vars(args), 
                   'prfs': {}}
    
    
    for epoch in range(args.num_epochs):
     
        train_scores = train(model, train_iterator, criterion, optimizer, metrics)
        valid_scores = evaluate(model, valid_iterator, criterion, metrics)        
        
        # Update output dictionary
        output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
        output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
                           
        # Save weights if is_best
        is_best_loss = valid_scores['loss'] < best_valid_loss
        is_best_f1 = valid_scores['f1'] > best_valid_f1
        
        if args.save_model == 'loss':
            utils.save_checkpoint({'epoch': epoch+1,
                                   'state_dict': model.state_dict(),
                                   'optim_Dict': optimizer.state_dict()},
                                   is_best = is_best_loss, checkdir = args.exp_dir)
        if args.save_model == 'f1':
            utils.save_checkpoint({'epoch': epoch+1,
                                   'state_dict': model.state_dict(),
                                   'optim_Dict': optimizer.state_dict()},
                                   is_best = is_best_f1, checkdir = args.exp_dir)
        
        if is_best_loss:
            best_valid_loss = valid_scores['loss']                    
            utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_loss.json'))
            
        if is_best_f1:
            best_valid_f1 = valid_scores['f1']                    
            utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_f1.json'))
        
        # Save the latest valid scores in exp_dir
        # utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'last_val_scores.json'))

        print("\nEpoch {}/{}...".format(epoch+1, args.num_epochs))                       
        print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
        print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
    
    # Write performance and args to json
    prfs_name = os.path.basename(args.exp_dir)+'.json'
    with open(os.path.join(args.exp_dir, prfs_name), 'w') as fout:
        json.dump(output_dict, fout, indent=4)
                   
    

        
def test(model, test_iterator, criterion, metrics, args, restore_file):   
     
    utils.load_checkpoint(os.path.join(args.exp_dir, restore_file + '.pth.tar'), model)
    test_scores = evaluate(model, test_iterator, criterion, metrics)
    # save_path = os.path.join(args.exp_dir, "test_scores.json")
    # utils.save_dict_to_json(test_scores, save_path)  
    print('\n[Test] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | recall: {3:.2f}% | precision: {4:.2f}% | specificity: {5:.2f}%'.format(
            test_scores['loss'], test_scores['accuracy']*100, test_scores['f1']*100, test_scores['recall']*100, test_scores['precision']*100, test_scores['specificity']*100))
    
    return test_scores

