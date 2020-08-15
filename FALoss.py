# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 19:06:38 2020

@author: 三岁嗷嗷帅
"""

import torch
import torch.nn as nn



class FALoss(nn.Module):
    def __init__(self):
        super(FALoss,self).__init__()
    def get_sim(self,fea):
        
        b,c,h,w = fea.shape
        fea = fea.view(b,c,-1)  
        fea = fea.permute((0,2,1)) 
        fea_norm = torch.norm(fea,p=2,dim=2) 
        fea = (fea.permute(2,1,0) / (fea_norm.permute(1,0) + 1e-6)).permute(2,1,0)
        sim = torch.matmul(fea,fea.permute(0,2,1))
        return sim
    def forward(self,source_fea,target_fea):
        source_sim =  self.get_sim(source_fea)
        target_sim = self.get_sim(target_fea)
        FALoss = torch.mean(torch.abs(target_sim-source_sim))
        return FALoss


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)
        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)


        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))


        return diff_loss

if __name__ == '__main__':
   pass
    
        