# ------------------------------------------------------------------------
# 3D Deformable Self-attention
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

def ms_deform_attn_core_pytorch_3D(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    #print("#######value.shape#######",value.shape)
    #print("#####attention_weights####",attention_weights.shape)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape  #head level points
    #print("#######sampling_locations.shape#######",sampling_locations.shape)
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    #print("#######value_list#######",len(value_list))
    sampling_grids = 2 * sampling_locations - 1
    #print("######sampling_grids#######",sampling_grids.shape)
    # sampling_grids = 3 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        #print("lid_",lid_)
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        #print("#####value_l_####",value_l_.shape)
        #print("###sampling_grid_l_lids######",sampling_grids[:, :, :, lid_].shape)
        #print("###sampling_grid_l_lids2######",sampling_grids[:, :, :,].shape)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)[:,:,:,:]
        #print("#####sampling_grid_l_####",sampling_grid_l_.shape)
        #print("******F*****",F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False).shape)
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_.to(dtype=value_l_.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)[:,:,:]
        #print("#####sampling_value_l_####",sampling_value_l_.shape)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    #print("#####attention_weights####",attention_weights.shape)
    #print("#####torch.stack(sampling_value_list, dim=-2)####",torch.stack(sampling_value_list, dim=-2).shape)
    #print("#####torch.stack(sampling_value_list, dim=-2).flatten(-2)####",(torch.stack(sampling_value_list, dim=-2).flatten(-2)).shape)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    #print("#####output####",output.shape)
    return output.transpose(1, 2).contiguous()