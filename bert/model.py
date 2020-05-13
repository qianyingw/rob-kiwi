#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 14:56:39 2020

@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertConfig, BertModel
from transformers import AlbertPreTrainedModel, AlbertConfig, AlbertModel

#%%
class BertLinear(BertPreTrainedModel):

    def __init__(self, bert_config: BertConfig):
        super().__init__(bert_config)
        
        self.bert = BertModel(bert_config)
        
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.fc = nn.Linear(bert_config.hidden_size, bert_config.num_labels)
        self.fc_bn = nn.BatchNorm1d(bert_config.num_labels)
        # self.fc = nn.Linear(bert_config.hidden_size * bert_config.n_chunks, bert_config.num_labels)
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if bert_config.unfreeze == "embed":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "embed_enc0":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name:
                    param.requires_grad = True
                    
        if bert_config.unfreeze == "embed_enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "enc0":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True
        
        if bert_config.unfreeze == "embed_pooler":
            for name, param in self.bert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "pooler":
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "enc-1":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name:
                    param.requires_grad = True
        
        if bert_config.unfreeze == "enc-1_pooler":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name or "pooler" in name:
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
        out = self.fc_bn(out)
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
        self.fc_bn = nn.BatchNorm1d(bert_config.num_labels)
        self.tanh = nn.Tanh()
        self.init_weights()
        
        # Default: freeze bert
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if bert_config.unfreeze == "embed":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "embed_enc0":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name:
                    param.requires_grad = True
                    
        if bert_config.unfreeze == "embed_enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "embeddings" in name or "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "enc0":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "enc0_pooler":
            for name, param in self.bert.named_parameters():
                if "encoder.layer.0" in name or "pooler" in name:
                    param.requires_grad = True
        
        if bert_config.unfreeze == "embed_pooler":
            for name, param in self.bert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if bert_config.unfreeze == "pooler":
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True
                    
        if bert_config.unfreeze == "enc-1":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name:
                    param.requires_grad = True
        
        if bert_config.unfreeze == "enc-1_pooler":
            n_layer = sum([1 for name, _ in self.bert.named_parameters() if "encoder.layer" in name])
            last_layer = "encoder.layer." + str(int(n_layer/16-1))  # each enc layer has 16 pars
            for name, param in self.bert.named_parameters():               
                if last_layer in name or "pooler" in name:
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
            # Output of BertModel: (last_hidden_state, pooler_output, hidden_states, attentions)
            # Last layer hidden-state of the first token of the sequence (classification token)
            pool_i = self.bert(input_ids = doc[i+1,:,0], 
                               attention_mask = doc[i+1,:,1], 
                               token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
            
        
        dp = self.dropout(pooled)  # [batch_size, num_chunks, bert_hidden_size]
        # output: [batch_size, num_chunks, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        
        # h_n = output[:,-1,].squeeze(1)  # [batch_size, hidden_size]
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]
        
        out = self.fc(h_n)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        # out = self.tanh(out)   # [batch_size, num_labels]
        
        return out


#%%
class AlbertLinear(AlbertPreTrainedModel):

    def __init__(self, albert_config: AlbertConfig):
        super().__init__(albert_config)
        
        self.albert = AlbertModel(albert_config)
        
        self.dropout = nn.Dropout(albert_config.hidden_dropout_prob)
        self.fc = nn.Linear(albert_config.hidden_size, albert_config.num_labels)
        self.fc_bn = nn.BatchNorm1d(albert_config.num_labels)
        # self.fc = nn.Linear(albert_config.hidden_size * albert_config.n_chunks, albert_config.num_labels)
        self.init_weights()
        
        # Default: freeze albert
        for name, param in self.albert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if albert_config.unfreeze == "embed":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "embed_enc0":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name or "encoder" in name:
                    param.requires_grad = True
                    
        if albert_config.unfreeze == "embed_enc0_pooler":
            for name, param in self.albert.named_parameters():
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "enc0":
            for name, param in self.albert.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "enc0_pooler":
            for name, param in self.albert.named_parameters():
                if "encoder" in name or "pooler" in name:
                    param.requires_grad = True
        
        if albert_config.unfreeze == "embed_pooler":
            for name, param in self.albert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "pooler":
            for name, param in self.albert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True
        
     
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.albert(input_ids = doc[0,:,0], 
                             attention_mask = doc[0,:,1], 
                             token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        for i in range(batch_size-1):
            pool_i = self.albert(input_ids = doc[i+1,:,0], 
                                 attention_mask = doc[i+1,:,1], 
                                 token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
 
                
        dp = self.dropout(pooled)  # [batch_size, num_chunks, hidden_size]  
        # concat = dp.view(batch_size, -1)  # [batch_size, num_chunks*hidden_size]
        if self.albert.config.linear_max == True:
            dp = torch.max(dp, dim=1).values  # [batch_size, hidden_size]
        else:
            dp = torch.mean(dp, dim=1)  # [batch_size, hidden_size]
        # dp = dp.sum(dim=1) # [batch_size, hidden_size]

        out = self.fc(dp)  # [batch_size, num_labels]   
        out = self.fc_bn(out)
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
             
        return out

#%%
class AlbertLSTM(AlbertPreTrainedModel):

    def __init__(self, albert_config: AlbertConfig):
        super().__init__(albert_config)
        
        self.albert = AlbertModel(albert_config)
        
        self.dropout = nn.Dropout(albert_config.hidden_dropout_prob)
                
        self.lstm = nn.LSTM(input_size = albert_config.hidden_size, hidden_size = albert_config.hidden_size,
                            num_layers = 1, dropout = 0, 
                            batch_first = True, bidirectional = False)
        
        self.fc = nn.Linear(albert_config.hidden_size, albert_config.num_labels)
        self.fc_bn = nn.BatchNorm1d(albert_config.num_labels)
        self.tanh = nn.Tanh()
        self.init_weights()  
        
        # Default: freeze albert
        for name, param in self.albert.named_parameters():
            param.requires_grad = False  

        # Unfreeze layers
        if albert_config.unfreeze == "embed":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "embed_enc0":
            for name, param in self.albert.named_parameters():
                if "embeddings" in name or "encoder" in name:
                    param.requires_grad = True
                    
        if albert_config.unfreeze == "embed_enc0_pooler":
            for name, param in self.albert.named_parameters():
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "enc0":
            for name, param in self.albert.named_parameters():
                if "encoder" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "enc0_pooler":
            for name, param in self.albert.named_parameters():
                if "encoder" in name or "pooler" in name:
                    param.requires_grad = True
        
        if albert_config.unfreeze == "embed_pooler":
            for name, param in self.albert.named_parameters():
                if "embed" in name or "pooler" in name:
                    param.requires_grad = True 
                    
        if albert_config.unfreeze == "pooler":
            for name, param in self.albert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = True
                                   
    
    def forward(self, doc):
        """
        Input:
            doc: [batch_size, num_chunks, 3, max_chunk_len]            
        Returns:
            out: [batch_size, output_dim]       
            
        """    
        batch_size = doc.shape[0]        
        
        pooled = self.albert(input_ids = doc[0,:,0], 
                             attention_mask = doc[0,:,1], 
                             token_type_ids = doc[0,:,2])[1].unsqueeze(0) 
        
        for i in range(batch_size-1):
            # Output of BertModel: (last_hidden_state, pooler_output, hidden_states, attentions)
            # Last layer hidden-state of the first token of the sequence (classification token)
            pool_i = self.albert(input_ids = doc[i+1,:,0], 
                                 attention_mask = doc[i+1,:,1], 
                                 token_type_ids = doc[i+1,:,2])[1]
            pooled = torch.cat((pooled, pool_i.unsqueeze(0)), dim=0)
            
        
        dp = self.dropout(pooled)  # [batch_size, num_chunks, bert_hidden_size]
        # output: [batch_size, num_chunks, n_directions*hidden_size], output features from last layer for each t
        # h_n: [n_layers*n_directions, batch_size, hidden_size], hidden state for t=seq_len
        # c_n: [n_layers*n_directions, batch_size, hidden_size], cell state fir t=seq_len
        output, (h_n, c_n) = self.lstm(dp)
        
        
        # h_n = output[:,-1,].squeeze(1)  # [batch_size, hidden_size]
        h_n = h_n.squeeze(0)  # [batch_size, hidden_size]
        
        out = self.fc(h_n)  # [batch_size, num_labels]  
        out = self.fc_bn(out)  
        out = F.softmax(out, dim=1)  # [batch_size, num_labels]
        # out = self.tanh(out)   # [batch_size, num_labels]
        
        return out