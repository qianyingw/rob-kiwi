#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 20:23:39 2020

@author: qwang
"""

from transformers import XLNetConfig, XLNetPreTrainedModel, XLNetTokenizer, XLNetModel

import torch
import torch.nn as nn

from hgf.modeling_utils import SequenceSummary

text = "Hello, my dog is cute"
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
tokens = tokenizer.tokenize(text, add_special_tokens=False)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

model = XLNetModel.from_pretrained('xlnet-base-cased')
outputs = model(torch.tensor(token_ids).unsqueeze(0))
last_hidden_states = outputs[0] 



#%%
class XLNetLSTM(XLNetPreTrainedModel):
    
    def __init__(self, config: XLNetConfig):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.xlnet = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        
        
        self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(config.hidden_size*3, config.num_labels)
        self.fc_bn = nn.BatchNorm1d(config.num_labels)

        self.init_weights()


    def forward(self, doc):
        """     
        Input:
            doc: [batch_size, ??, ?, ??]  [batch_size, seq_len, 3, max_chunk_len]           
        Returns:
            out: [batch_size, output_dim]  

        """
        
        pooled = self.bert(input_ids = doc[0,:,0], 
                           attention_mask = doc[0,:,1], 
                           token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        
        xln_outputs = self.xlnet(input_ids, mems=mems, 
                                           head_mask=head_mask, 
                                               inputs_embeds=inputs_embeds)


        output = transformer_outputs[0]

        output = self.sequence_summary(output)
        logits = self.logits_proj(output)

        outputs = (logits,) + transformer_outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # return (loss), logits, (mems), (hidden states), (attentions)