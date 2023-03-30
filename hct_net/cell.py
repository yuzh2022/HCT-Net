import torch
import torch.nn as nn

from operations import *
from genotypes import CellLinkDownPos,CellLinkUpPos,CellPos


class MixedOp (nn.Module):
    def __init__(self, c, stride,mixop_type,switch,dp):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_type = mixop_type
        #print("switch:",len(switch))
        if mixop_type=="down":
            PRIMITIVES=CellLinkDownPos
            #print("switch:", len(switch))
            assert stride==2 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "
        elif mixop_type=='up':
            PRIMITIVES=CellLinkUpPos
            assert stride==2 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "
        else:
            PRIMITIVES=CellPos
            assert stride==1 and len(PRIMITIVES)==len(switch),"the mixop type or nums is wrong "

        for i in range(len(switch)):
            if switch[i]:
                primitive = PRIMITIVES[i]
                op = OPS[primitive](c, stride, affine=False, dp=dp)
                self._ops.append(op)


    def forward(self, x, weight_normal, weight_down,weight_up,pos_normal,pos_up,pos_down):
        # weight_normal: M * 1 where M is the number of normal primitive operations
        # weight_up_or_down: K * 1 where K is the number of up or down primitive operations
        # we have three different weights

        if self._op_type == 'down':
            rst=weight_down[pos_down]*self._ops[pos_down](x)
            #rst = sum(w * op(x) for w, op in zip(weight_down, self._ops))
        elif self._op_type=="up":
            rst = weight_up[pos_up] * self._ops[pos_up](x)
            #rst = sum(w * op(x) for w, op in zip(weight_up, self._ops))
        else:
            rst = weight_normal[pos_normal] * self._ops[pos_normal](x)
            #rst = sum(w * op(x) for w, op in zip(weight_normal, self._ops))
        return rst


class Cell(nn.Module):
    #c, stride, mixop_type, switch, dropout_prob
    def __init__(self, meta_node_num, c_prev_prev, c_prev, c,
                 switch_normal,switch_down,switch_up,cell_type,dp=0):
        super(Cell, self).__init__()
        self.c_prev_prev = c_prev_prev #-1
        self.c_prev = c_prev #16
        self.c = c #32
        self._meta_node_num = meta_node_num #4
        self._multiplier = meta_node_num #4
        self._input_node_num = 2
        self._steps=meta_node_num
        #print("swich_normal:",len(switch_up))
        #print("1:---",self.c_prev_prev,self.c_prev,self.c)
        # the sanme feature map size (ck-2)
        if c_prev_prev !=-1:
            self.preprocess0=ConvOps(c_prev_prev,c,kernel_size=1, affine=False, ops_order='act_weight_norm')
        else:
            self.preprocess0=None
        # must be exits!
        self.preprocess1=ConvOps(c_prev,c,kernel_size=1, affine=False, ops_order='act_weight_norm') #64 ,32,1
        #print("1:",self.preprocess1)
        '''1: ConvOps(
              (norm): GroupNorm(2, 32, eps=1e-05, affine=False)
              (activation): ReLU()
              (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            )'''
        self._ops = nn.ModuleList()
        # cell_type=normal_down,normal_normal,normal_up ,three types
        # c, stride, mixop_type, switch, dropout_prob
        if cell_type=="normal_normal":
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        op=MixedOp(c,stride=1,mixop_type='normal',switch=switch_normal[switch_count],dp=dp)
                    self._ops.append(op)
                    #print("swich_normal:", len(switch_normal))
                    switch_count+=1
        # input node 0 is normal,1 is down op /up op
        elif cell_type=='normal_down':
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        if j==1:
                            op = MixedOp(c, stride=2, mixop_type='down', switch=switch_down[switch_count],dp=dp)
                        else:
                            op = MixedOp(c, stride=1, mixop_type='normal', switch=switch_normal[switch_count], dp=dp)

                    self._ops.append(op)
                    switch_count+=1

        elif cell_type=='normal_up':
            switch_count=0
            for i in range(self._meta_node_num):
                for j in range(self._input_node_num+i):
                    # the first input node is not exists!
                    if c_prev_prev==-1 and j==0:
                        op=None
                    else:
                        if j == 1:
                            op = MixedOp(c, stride=2, mixop_type='up', switch=switch_up[switch_count], dp=dp)
                        else:
                            op = MixedOp(c, stride=1, mixop_type='normal', switch=switch_normal[switch_count], dp=dp)
                    self._ops.append(op)
                    switch_count+=1


    def forward(self, s0, s1, weights_normal,weights_down,weights_up):

        if s0 is not None :
            s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        offset = 0
        switch=0
        for i in range(self._steps):
            tmp_list=[]
            for j, h in enumerate(states):
                if h is not None:
                    pos_normal = (weights_normal[switch, :] == 1).nonzero().item()
                    pos_up = (weights_up[switch, :] == 1).nonzero().item()
                    pos_down = (weights_down[switch, :] == 1).nonzero().item()
                    tmp_list.append(self._ops[offset+j](h,weights_normal[offset+j],weights_down[offset+j],weights_up[offset+j],pos_normal,pos_up,pos_down))
                    switch+=1
            s = sum(consistent_dim(tmp_list))
            offset += len(states)
            states.append(s)
        concat_feature = torch.cat(states[-self._multiplier:], dim=1)
        #return  self.ReLUConvBN (concat_feature)
        return concat_feature

