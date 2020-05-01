#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:56:39 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel


#%%
class BertLinear(BertPreTrainedModel):

    def __init__(self, bert_config: BertConfig):
        super().__init__(bert_config)
        
        self.bert = BertModel(bert_config)
        
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.fc = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        # self.fc = nn.Linear(bert_config.hidden_size * bert_config.n_chunks, bert_config.num_labels)
        # self.init_weights()
        
        # Freeze bert
        if bert_config.freeze_bert == True:
            for param in self.bert.parameters():
                param.requires_grad = False 
                
            # Unfreeze last encoder layer
            if bert_config.unfreeze_layer == 0:
                for name, param in self.bert.named_parameters():
                    if "pooler" in name:
                        param.requires_grad = True
            else:
                layer_name = "encoder.layer." + str(bert_config.unfreeze_layer-1)
                for name, param in self.bert.named_parameters():
                    if layer_name in name or "pooler" in name:
                        param.requires_grad = True
     
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.bert(input_ids = doc[0,:,0], 
                           attention_mask = doc[0,:,1], 
                           token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        for i in range(batch_size-1):
            pool_i = self.bert(input_ids = doc[i+1,:,0], 
                               attention_mask = doc[i+1,:,1], 
                               token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
 
                
        dp = self.dropout(pooled)  # [batch_size, num_chunks, hidden_size]  
        # concat = dp.view(batch_size, -1)  # [batch_size, num_chunks*hidden_size]
        if self.bert.config.linear_max == True:
            dp = torch.max(dp, dim=1).values  # [batch_size, hidden_size]
        else:
            dp = torch.mean(dp, dim=1)  # [batch_size, hidden_size]
        # dp = dp.sum(dim=1) # [batch_size, hidden_size]

        out = self.fc(dp)  # [batch_size, num_labels]         
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
             
        return out


#%%
class BertLSTM(BertPreTrainedModel):

    def __init__(self, bert_config: BertConfig):
        super().__init__(bert_config)
        
        self.bert = BertModel(bert_config)
        
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
                
        self.lstm = nn.LSTM(input_size = bert_config.hidden_size, hidden_size = bert_config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        # self.init_weights()
    
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        num_chunks = doc.shape[1]
        
        pooled = torch.zeros((batch_size, num_chunks, self.bert.config.hidden_size), dtype=torch.float)
        for i in range(batch_size):
            # Output of BertModel: (last_hidden_state, pooler_output, hidden_states, attentions)
            # Last layer hidden-state of the first token of the sequence (classification token)
            pooled[i] = self.bert(input_ids = doc[i,:,0], 
                                  attention_mask = doc[i,:,1], 
                                  token_type_ids = doc[i,:,2])[1]
        
        
        dp = self.dropout(pooled)  # [batch_size, num_chunks, bert_hidden_size]
        # output: [batch_size, num_chunks, n_directions*hidden_size], output features from last layer for each t
        # h_n: [batch_size, n_layers*n_directions, hidden_size], hidden state for t=seq_len
        # c_n: [batch_size, n_layers*n_directions, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        h_n = h_n.squeeze(1)  # [batch_size, hidden_size]
        
        out = self.fc(h_n)  # [batch_size, num_labels]         
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        
        return out