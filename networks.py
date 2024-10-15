# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.distributed as dist
import math

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)
        
        
class Transformers_Encoder_Classifier(nn.Module):

    #Transformers Encoder Neural Network for a Classifier

    def __init__(self, batchsize, slide_N, prepro_N, Overlap_rate, input_shape, num_heads, headsize, ff_dim, num_transformer_block, mlp_units,
     drop, mlp_drop, n_class, epochs,flag,d_model,self_attn_numhead,sub_factor,Compact_L):
        super().__init__()

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                position = torch.arange(max_len).unsqueeze(1)
                div_term1 = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
                div_term2 = torch.exp(torch.arange(1, d_model, 2) * (-math.log(10000.0) / d_model))
                pe = torch.zeros(max_len, 1, d_model)
                pe[:, 0, 0::2] = torch.sin(position * div_term1)
                pe[:, 0, 1::2] = torch.cos(position * div_term2)
                self.register_buffer('pe', pe)

            def forward(self, x):
                """
                Arguments:
                    x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
                """
                x = torch.permute(x, (1,0,2))
                x = x + self.pe[:x.size(0)]
                x = torch.permute(x, (1,0,2))
                return self.dropout(x)

        self.batchsize = batchsize
        self.slide_N = slide_N
        self.prepro_N = prepro_N
        self.Overlap_rate = Overlap_rate
        self.input_shape = input_shape
        self.num_heads = num_heads
        self.headsize = headsize
        self.ff_dim = ff_dim
        self.num_transformer_block = num_transformer_block
        self.mlp_units = mlp_units
        self.drop = drop
        self.mlp_drop = mlp_drop
        self.n_class = n_class
        self.epochs = epochs
        self.d_model = d_model
        self.self_attn_numhead = self_attn_numhead
        self.flag = flag
        self.Compact_L = Compact_L
        self.L1 = 5000
        self.L2 = 5000*self.slide_N
        self.factor = sub_factor
        self.factor2 = 16
        self.final_seq = self.input_shape[1]*int(self.Compact_L/self.factor2/self.factor2)*self.slide_N

        # Self-attention with shared weights among sub-stack
        self.self_attention = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.self_attn_numhead, batch_first=True)           
        # MLP with shared weights among substacks
        self.MLP = nn.Sequential(
            #nn.Linear(self.L1, self.L1/2),
            #nn.Dropout(self.drop),
            #nn.ReLU(inplace=True),
            nn.Linear(int(self.L1), self.Compact_L, bias=False),
        )

        # Positional encoding for each slide window
        self.pos_encoder = PositionalEncoding(self.d_model, 0, self.L1)
        # Shared transformer + MLP  weights along slide windows
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads, dim_feedforward =self.ff_dim, dropout =self.drop, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_transformer_block)
        # MLP in each slide window
        self.MLP_sup1 = nn.Sequential(
            nn.Linear(self.Compact_L, int(self.Compact_L/self.factor2), bias=False),
            #nn.LayerNorm([self.input_shape[1], int(self.L3/self.factor2)]),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.Compact_L/self.factor2), int(self.Compact_L/self.factor2/self.factor2), bias=False),
            #nn.LayerNorm([self.input_shape[1], int(self.L3/self.factor2/self.factor2)]),
            nn.Dropout(self.drop),
            nn.ReLU(inplace=True),
            nn.Flatten(1,2),
        )
        self.MLP_sup2 = nn.Sequential(
            nn.Linear(self.input_shape[1]*int(self.Compact_L/self.factor2/self.factor2), self.mlp_units[0]),
            nn.LayerNorm([self.mlp_units[0],]),
            nn.Dropout(self.mlp_drop),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_units[0], self.mlp_units[1]),
            nn.LayerNorm([self.mlp_units[1],]),
            nn.Dropout(self.mlp_drop),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_units[1], self.n_class),
            nn.Softmax(dim=1),
        )
        # MLP before the last classification output
        self.MLP_final = nn.Sequential(
            nn.Linear(self.final_seq, int(self.final_seq/self.factor)),
            nn.LayerNorm([int(self.final_seq/self.factor),]),
            nn.Dropout(self.mlp_drop),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.final_seq/self.factor), int(self.final_seq/self.factor/self.factor)),
            nn.LayerNorm([int(self.final_seq/self.factor/self.factor),]),
            nn.Dropout(self.mlp_drop),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.final_seq/self.factor/self.factor), int(self.final_seq/self.factor/self.factor/self.factor)),
            nn.LayerNorm([int(self.final_seq/self.factor/self.factor/self.factor),]),
            nn.Dropout(self.mlp_drop),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.final_seq/self.factor/self.factor/self.factor), self.n_class, bias=False),           
        )
        self.Softmax = nn.Sequential(nn.Softmax(dim=1),)
    
        


    def forward(self, inputs):
        if self.flag == 1:   
            print('Shape of the input: ',inputs.shape)                                              
        # Length of sub-stack
        # Self attention
        inputs_temp = inputs[:,0:self.L1,:] 
        #print("Shape inputs_temp",inputs_temp.shape)
        #print("Shape L1",self.L1)  
        # MLP
        x_self_attention_total = torch.permute(inputs_temp, (0, 2, 1))           
        x_MLP = self.MLP(x_self_attention_total)
        #print("Shape x_MLP",x_MLP.shape)
        x_MLP = torch.permute(x_MLP, (0, 2, 1))                         
        x_MLP_total = self.pos_encoder(x_MLP)
        x_Trans_Encoder_MLP = self.transformer_encoder(x_MLP_total)
        #print("x_Trans_Encoder_MLP.shape",x_Trans_Encoder_MLP.shape)
        x_Trans_Encoder_MLP = torch.permute(x_Trans_Encoder_MLP, (0, 2, 1))
        #print(x_Trans_Encoder_MLP.shape)
        x_Trans_Encoder_MLP_total1 = self.MLP_sup1(x_Trans_Encoder_MLP)
        x_Trans_Encoder_MLP2 = self.MLP_sup2(x_Trans_Encoder_MLP_total1)
        x_Trans_Encoder_MLP_total2 = torch.unsqueeze(x_Trans_Encoder_MLP2,2)
        #print('Shape x_Trans_Encoder_MLP_total1',x_Trans_Encoder_MLP_total1.shape)
        for j in range(self.slide_N-1):
            #inputs_temp = inputs[:,(j+2)*math.floor(inputs.shape[0]/self.slide_N)-self.L1:(j+2)*math.floor(inputs.shape[0]/self.slide_N),:]
            inputs_temp = inputs[:,(j+2)*self.L1-self.L1:(j+2)*self.L1:]
            x_self_attention_total = torch.permute(inputs_temp, (0, 2, 1))           
            x_MLP = self.MLP(x_self_attention_total)
            x_MLP = torch.permute(x_MLP, (0, 2, 1))                       
            x_MLP = self.pos_encoder(x_MLP)
            x_Trans_Encoder_MLP = self.transformer_encoder(x_MLP)
            x_Trans_Encoder_MLP = torch.permute(x_Trans_Encoder_MLP, (0, 2, 1))
            x_Trans_Encoder_MLP = self.MLP_sup1(x_Trans_Encoder_MLP)
            x_Trans_Encoder_MLP2 = self.MLP_sup2(x_Trans_Encoder_MLP)
            x_Trans_Encoder_MLP_total2 = torch.cat((x_Trans_Encoder_MLP_total2,torch.unsqueeze(x_Trans_Encoder_MLP2,2)), axis=2) 
            x_Trans_Encoder_MLP_total1 = torch.cat((x_Trans_Encoder_MLP_total1,x_Trans_Encoder_MLP), axis=1)
        #print('Shape x_Trans_Encoder_MLP_total1',x_Trans_Encoder_MLP_total1.shape) 
        #print('Shape x_Trans_Encoder_MLP_total2',x_Trans_Encoder_MLP_total2.shape) 
        x_Trans_Encoder_MLP_class = self.MLP_final(x_Trans_Encoder_MLP_total1)   
        #x_Trans_Encoder_MLP_class = torch.squeeze(x_Trans_Encoder_MLP_class,2)
        x_Trans_Encoder_MLP_class = self.Softmax(x_Trans_Encoder_MLP_class)
        if self.flag == 1:  
            print('Shape of the classification output: ',x_Trans_Encoder_MLP_class.shape)
        return x_Trans_Encoder_MLP_total2, x_Trans_Encoder_MLP_class
        