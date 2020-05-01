#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:52:33 2020

@author: qwang
"""

import os
import json
from tqdm import tqdm
import torch
import torch.nn as nn

import utils

#%% Train

def train(model, data_loader, optimizer, scheduler, criterion, metrics, device, clip, accum_step, threshold=0.5):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    
    model.train()
    
    optimizer.zero_grad()
    with tqdm(total=len_iter) as progress_bar:
        for i, batch in enumerate(data_loader):
            
            batch = tuple(t.to(device) for t in batch)           
            batch_doc, batch_label, batch_len = batch    
            
            preds = model(batch_doc)  # preds.shape = [batch_size, num_labels]
            
            loss = criterion(preds, batch_label)    
            scores['loss'] += loss.item()
            epoch_scores = metrics(preds, batch_label, threshold)  # dictionary of 5 metric scores
            for key, value in epoch_scores.items():               
                scores[key] += value  
            
            loss = loss / accum_step  # loss gradients are accumulated by loss.backward() so we need to ave accumulated loss gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # prevent exploding gradients
                      
            # Gradient accumulation    
            if (i+1) % accum_step == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            # Update progress bar                          
            progress_bar.update(1)  
    
    for key, value in scores.items():
        scores[key] = value / len_iter   
    return scores





#%% Evaluate   
def evaluate(model, data_loader, criterion, metrics, device, threshold=0.5):
    
    scores = {'loss': 0, 'accuracy': 0, 'f1': 0, 'recall': 0, 'precision': 0, 'specificity': 0}
    len_iter = len(data_loader)
    model.eval()

    with torch.no_grad():
        with tqdm(total=len_iter) as progress_bar:
            for batch in data_loader:
                
                batch = tuple(t.to(device) for t in batch)
                batch_doc, batch_label, batch_len = batch          
                preds = model(batch_doc)
                
                loss = criterion(preds, batch_label) 
                epoch_scores = metrics(preds, batch_label, threshold)
                
                scores['loss'] += loss.item()
                for key, value in epoch_scores.items():               
                    scores[key] += value        
                progress_bar.update(1)  # update progress bar   
                
    for key, value in scores.items():
        scores[key] = value / len_iter   

    return scores

#%% train_eval
    
def train_evaluate(model, train_iterator, valid_iterator, optimizer, scheduler, criterion, metrics, args, device, restore_file=None):
    """
    
    """
    
    if os.path.exists(args.exp_dir) == False:
        os.makedirs(args.exp_dir)     
    
    if restore_file is not None:
        restore_path = os.path.join(args.exp_dir, restore_file + '.pth.tar')
        utils.load_checkpoint(restore_path, model, optimizer)
    

    
    # Create args and output dictionary (for json output)
    output_dict = {'args': vars(args), 'prfs': {}}
    
    for epoch in range(args.num_epochs):   
        train_scores = train(model, train_iterator, optimizer, scheduler, criterion, metrics, device, args.clip, args.accum_step, args.threshold)
        valid_scores = evaluate(model, valid_iterator, criterion, metrics, device, args.threshold)        

        # Update output dictionary
        output_dict['prfs'][str('train_'+str(epoch+1))] = train_scores
        output_dict['prfs'][str('valid_'+str(epoch+1))] = valid_scores
        
   
#        is_best = (valid_scores['loss']-min_valid_loss <= args.stop_c1) and (max_valid_f1-valid_scores['f1'] <= args.stop_c2)
#        if is_best == True:       
#            utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'best_val_scores.json'))
#        
#        # Save model
#        if args.save_model == True:
#            utils.save_checkpoint({'epoch': epoch+1,
#                                   'state_dict': model.state_dict(),
#                                   'optim_Dict': optimizer.state_dict()},
#                                   is_best = is_best, checkdir = args.exp_dir)

        # Save the latest valid scores in exp_dir
        # utils.save_dict_to_json(valid_scores, os.path.join(args.exp_dir, 'last_val_scores.json'))

        print("\n\nEpoch {}/{}...".format(epoch+1, args.num_epochs))                       
        print('\n[Train] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%'.format(
            train_scores['loss'], train_scores['accuracy']*100, train_scores['f1']*100, train_scores['recall']*100, train_scores['precision']*100, train_scores['specificity']*100))
        print('[Val] loss: {0:.3f} | acc: {1:.2f}% | f1: {2:.2f}% | rec: {3:.2f}% | prec: {4:.2f}% | spec: {5:.2f}%\n'.format(
            valid_scores['loss'], valid_scores['accuracy']*100, valid_scores['f1']*100, valid_scores['recall']*100, valid_scores['precision']*100, valid_scores['specificity']*100))
        
    # Write performance and args to json
    prfs_name = os.path.basename(args.exp_dir)+'_prfs.json'
    prfs_path = os.path.join(args.exp_dir, prfs_name)
    with open(prfs_path, 'w') as fout:
        json.dump(output_dict, fout, indent=4)
        
    # Save performance plot    
    # utils.plot_prfs(prfs_json_path=prfs_path)
    
