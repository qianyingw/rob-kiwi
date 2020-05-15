#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:31:07 2020

@author: qwang
"""

from transformers import TransfoXLConfig, TransfoXLPreTrainedModel, TransfoXLTokenizer, TransfoXLModel

import torch
import torch.nn as nn
import torch.nn.functional as F


import logging
logger = logging.getLogger(__name__)



tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
text = "Hello, my dog is cute"

# tokenizer.encode => tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
tokens = tokenizer.tokenize(text, add_special_tokens=True, add_space_before_punct_symbol=True)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

model = TransfoXLModel.from_pretrained('transfo-xl-wt103')

outputs = model(torch.tensor(token_ids).unsqueeze(0))
last_hidden_states, mems = outputs[:2]


#%%
class TransfoXLLSTM(TransfoXLPreTrainedModel):
    
    def __init__(self, config: TransfoXLConfig):
        super().__init__(config)
        
        self.transformer = TransfoXLModel(config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
                
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size*3, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()

    

    def forward(self, input_ids=None, mems=None, head_mask=None, inputs_embeds=None, labels=None):
        """     
        Input:
            doc: [batch_size, ??, ?, ??]  [batch_size, seq_len, 3, max_chunk_len]           
        Returns:
            out: [batch_size, output_dim]  

        """

        transfo_outputs = self.transformer(input_ids, mems=mems, 
                                               head_mask=head_mask, 
                                               inputs_embeds=inputs_embeds)
        


        last_hidden_states = transfo_outputs[0]  # [batch_size, seq_len, hidden_size]
               
        dp = self.dropout(last_hidden_states)  # [batch_size, seq_len, hidden_size]
        # output: [batch_size, seq_len, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        # Concat pooling
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]
        h_max = torch.max(output, dim=1).values  # [batch_size, hidden_size]
        h_mean = torch.mean(output, dim=1)  # [batch_size, hidden_size]
        out = torch.cat((h_n, h_max, h_mean), dim=1)  # [batch_size, hidden_size*3]
        
        out = self.fc(out)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]

        return out  

