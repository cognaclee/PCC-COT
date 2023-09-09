# -*- coding: utf-8 -*-
# @Time        : 27/06/2022 10:00 PM
# @Description :
# @Author      : li ze zeng
# @Email       : zezeng.lee@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.layer import EdgeConv

class PointAttention(nn.Module):
    def __init__(self, in_dim):
        super(PointAttention, self).__init__()
        layer = in_dim//4

        self.w_qs = nn.Conv1d(in_dim, layer, 1)
        self.w_ks = nn.Conv1d(in_dim, layer, 1)
        self.w_vs = nn.Conv1d(in_dim, in_dim, 1)
        self.gamma=nn.Parameter(torch.tensor(torch.zeros([1]))).cuda()

    def forward(self, inputs):
        q = self.w_qs(inputs)
        k = self.w_ks(inputs)
        v = self.w_vs(inputs)
        
        s = torch.matmul(q.transpose(2, 1), k)

        beta = F.softmax(s, dim=-1)  # attention map
        o = torch.matmul(v, beta)   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        x = self.gamma * o + inputs
        return x
        

class PointResNet(nn.Module):
    def __init__(self, in_dim,layers):
        super(PointResNet, self).__init__()
        
        chs = in_dim
        fea_dims = []
        for i in range(layers):
            fea_dims.append(chs)
            chs *= 2
        fea_dims.append(chs)
        for i in range(layers):
            fea_dims.append(chs)
            chs = chs //2
        #fea_dims.append(in_dim)   
        
        sequence = []
        for i in range(len(fea_dims)-1):
            sequence += [nn.Conv1d(fea_dims[i],fea_dims[i+1], 1),
            nn.GroupNorm(1,fea_dims[i+1]),
            nn.ReLU()]
        
        sequence += [nn.Conv1d(fea_dims[-1],in_dim, 1)]
        self.net = nn.Sequential(*sequence)

    def forward(self, x):
        res = self.net(x)
        out = x + res
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args
        layes = 2
        sequence = [nn.Conv1d(args.in_fdim,args.start_num, 1),
                    nn.GroupNorm(args.ngroups,args.start_num),
                    nn.ReLU()]
        chs = args.start_num
        '''sequence += [nn.Conv1d(chs,2*chs, 1),
            nn.GroupNorm(args.ngroups,2*chs),
            nn.ReLU()]
        chs *= 2
        '''
        for i in range(layes):
            sequence += [nn.Conv1d(chs,2*chs, 1),
            nn.GroupNorm(args.ngroups,2*chs),
            nn.ReLU()]
            chs *= 2
        #'''
        self.head_conv = nn.Sequential(*sequence)
        
        chs *= 2
        #self.attention_unit = PointAttention(chs)
        self.resnet_unit = PointResNet(chs,2)

        sequence = [nn.Conv1d(chs,chs, 1),nn.GroupNorm(args.ngroups,chs),nn.ReLU()]
        sequence += [nn.Conv1d(chs,2*chs, 1),nn.GroupNorm(args.ngroups,2*chs),nn.ReLU()]
        sequence += [nn.Conv1d(2*chs,chs, 1),nn.GroupNorm(args.ngroups,chs),nn.ReLU()]
        '''for i in range(layes):
            sequence += [nn.Conv1d(chs,2*chs, 1),
            nn.GroupNorm(args.ngroups,2*chs),
            nn.ReLU()]
            chs *= 2'''
        self.mid_conv = nn.Sequential(*sequence)
        sequence = [nn.Linear(chs,chs),nn.GroupNorm(args.ngroups,chs),nn.ReLU()]
        sequence += [nn.Linear(chs,1),nn.Sigmoid()]
        #sequence += [nn.Linear(chs,1)]
        self.tail_mlp = nn.Sequential(*sequence)

        


    def forward(self, inputs):
        features = self.head_conv(inputs)  
        features_global=torch.max(features,dim=2)[0] ##global feature
        global_fea = features_global.unsqueeze(2).repeat(1,1,features.shape[2])
   
        features = torch.cat([features, global_fea],dim=1)
        #features = self.attention_unit(features)
        features = self.resnet_unit(features)
        features = self.mid_conv(features)
        features=torch.max(features,dim=2)[0]
        outputs = self.tail_mlp(features).view(-1, 1)
        return outputs
