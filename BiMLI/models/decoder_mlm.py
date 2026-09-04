
import torch
import torch.nn as nn
from models.layer import *
# from models.model_util import activation_function_
    
class RotatE(nn.Module):
    def __init__(self, dim,dropout = 0.3):
        super(RotatE, self).__init__()
        self.dim = dim // 2  
        self.bn0 = nn.BatchNorm1d(dim)
        self.input_drop = nn.Dropout(dropout)

    def forward(self, e_embed, r_embed,ent_emd):
        h_embed = e_embed  
        h_embed = self.bn0(h_embed)
     
        h_embed = self.input_drop(h_embed)
     

        h_re, h_im = torch.chunk(h_embed, 2, dim=-1)

        r_phase = r_embed[:, :self.dim]  
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)

        h_r_re = h_re * r_re - h_im * r_im
        h_r_im = h_re * r_im + h_im * r_re
        h_embed= torch.cat([h_r_re, h_r_im], dim=-1)
        pred = torch.mm(h_embed, ent_emd.transpose(1, 0))
        # pred += self.bias.expand_as(pred)
        # pred = activation_function_(self.act_fun,pred)
        return pred
    

