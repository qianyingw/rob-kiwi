#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:23:39 2020

@author: qwang
"""

from transformers import XLNetConfig, XLNetPreTrainedModel, XLNetTokenizer, XLNetModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from hgf.modeling_utils import SequenceSummary

#%%
class XLNetLSTM(XLNetPreTrainedModel):
    
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        self.seq_summary = SequenceSummary(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()


    def forward(self, doc):
        """     
        Input:
            doc: [batch_size, seq_len, 2]  [batch_size, seq_len, 3, max_chunk_len]           
        Returns:
            out: [batch_size, output_dim]  

        """
        # input_ids / attnention_mask: [batch_size, seq_len]
        xln_out = self.xlnet(input_ids = doc[:,:,0], 
                             attention_mask = doc[:,:,1])
        
        last_layer_hidden = xln_out[0]  # [batch_size, seq_len, hidden_size]
        
        dp = self.dropout(last_layer_hidden)  # [batch_size, seq_len, hidden_size]
        # output: [batch_size, seq_len, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]. Or h_n = output[:,-1,].squeeze(1)
        out = h_n
        
        # Concat pooling
        # h_max = torch.max(output, dim=1).values  # [batch_size, hidden_size]
        # h_mean = torch.mean(output, dim=1)  # [batch_size, hidden_size]
        # out = torch.cat((h_n, h_max, h_mean), dim=1)  # [batch_size, hidden_size*3]
        
        out = self.fc(out)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        
        return out
        
    
#%%
class XLNetLinear(XLNetPreTrainedModel):
    
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        self.seq_summary = SequenceSummary(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()


    def forward(self, doc):
        """     
        Input:
            doc: [batch_size, seq_len, 2]           
        Returns:
            out: [batch_size, output_dim]  

        """
        # input_ids / attnention_mask: [batch_size, seq_len]
        xln_out = self.xlnet(input_ids = doc[:,:,0], 
                             attention_mask = doc[:,:,1])
        
        last_layer_hidden = xln_out[0]  # [batch_size, seq_len, hidden_size]

        # SequenceSummary computes a single vector summary of a sequence hidden states according to various possibilities:
        #    - 'last' => [default] take the last token hidden state (like XLNet)
        #    - 'first' => take the first token hidden state (like Bert)
        #    - 'mean' => take the mean of all tokens hidden states
        #    - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
        seq_sum = self.seq_summary(last_layer_hidden)  # [batch_size, hidden_size]
        
        dp = self.dropout(seq_sum)  # [batch_size, hidden_size]
        
        out = self.fc(dp)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]

        return out


#%%
class XLNetConv(XLNetPreTrainedModel):
    
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        self.seq_summary = SequenceSummary(config)
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1,
                                              out_channels = config.n_filters,
                                              kernel_size = (fsize, config.hidden_size)) for fsize in config.filter_sizes])   
          
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()


    def forward(self, doc):
        """     
        Input:
            doc: [batch_size, seq_len, 2]           
        Returns:
            out: [batch_size, output_dim]  

        """
        # input_ids / attnention_mask: [batch_size, seq_len]
        xln_out = self.xlnet(input_ids = doc[:,:,0], 
                             attention_mask = doc[:,:,1])
        
        xln = xln_out[0]  # [batch_size, seq_len, hidden_size]
        
        xln = xln.unsqueeze(1)  # [batch_size, 1, seq_len, hidden_size]
        
        
        conved = [F.relu(conv(xln)) for conv in self.convs]  # [batch_size, n_filters, (seq_len-fsize+1), 1]
        conved = [conv.squeeze(3) for conv in conved]  # [batch_size, n_filters, (seq_len-fsize+1)]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]  # [batch_size, n_filters, 1]
        pooled = [pool.squeeze(2) for pool in pooled]  # [batch_size, n_filters]
        
        cat = torch.cat(pooled, dim=1)  # [batch_size, n_filters * len(filter_sizes)]
        dp = self.dropout(cat)
        out = self.fc(dp)  # # [batch_size, output_dim]
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, output_dim]  

        return out