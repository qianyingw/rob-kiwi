#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 12:56:42 2020

@author: qwang
"""

from transformers import LongformerConfig, LongformerTokenizer, LongformerModel
from transformers import BertPreTrainedModel


import torch
import torch.nn as nn
import torch.nn.functional as F


#%%

class LongformerLinear(BertPreTrainedModel):
    config_class = LongformerConfig
    base_model_prefix = "longformer"

    def __init__(self, config: LongformerTokenizer):
        super().__init__(config)

        self.longformer = LongformerModel(config)
    
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.longformer.named_parameters():
            param.requires_grad = False  

    
    # @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    # @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="allenai/longformer-base-4096")
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, seq_len, 3]     
                 n_chunks is the number of chunks within the batch (same for each doc after PadDoc)
        Returns:
            out: [batch_size, output_dim]          

        """

        # if global_attention_mask is None:
        #     global_attention_mask = torch.zeros_like(input_ids)
        #     # global attention on cls token
        #     global_attention_mask[:, 0] = 1
        
         # input_ids / attnention_mask: [batch_size, seq_len]
        longformer_output = self.longformer(input_ids = doc[:,:,0],
                                            attention_mask = doc[:,:,1],
                                            global_attention_mask=doc[:,:,2])
        
        hidden_states = longformer_output[1]  # [batch_size, hidden_size]

        dp = self.dropout(hidden_states)  # [batch_size, hidden_size]
        
        out = self.fc(dp)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]


        return out