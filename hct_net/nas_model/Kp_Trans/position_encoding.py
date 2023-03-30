"""
Positional encodings for the transformer.
"""
import math
import torch
from torch import nn
from typing import Optional
from torch import Tensor

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=[32, 32], temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        #print("###x###",x.shape)
        bs, c, h, w = x.shape
        '''N_steps = c // 2
        if (c % 2) != 0:
            self.num_pos_feats = [N_steps, N_steps + c % 2]
        else:
            self.num_pos_feats = [N_steps, N_steps]'''
        mask = torch.zeros(bs, h, w, dtype=torch.bool).cuda()
        assert mask is not None
        not_mask = ~mask
        #d_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        #print("####y_embed#####",y_embed.shape)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        #print("####x_embed#####",x_embed.shape)
        if self.normalize:
            eps = 1e-6
            #d_embed = (d_embed - 0.5) / (d_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale
        #print("####y_embed#####",y_embed.shape)
        #print("####x_embed#####",x_embed.shape)
        dim_tx = torch.arange(self.num_pos_feats[0], dtype=torch.float32, device=x.device)
        dim_tx = self.temperature ** (3 * (dim_tx // 3) / self.num_pos_feats[0])

        dim_ty = torch.arange(self.num_pos_feats[1], dtype=torch.float32, device=x.device)
        dim_ty = self.temperature ** (3 * (dim_ty // 3) / self.num_pos_feats[1])


        #print("####dim_ty#####",dim_ty.shape)
        #print("####dim_tx#####",dim_tx.shape)

        #dim_td = torch.arange(self.num_pos_feats[2], dtype=torch.float32, device=x.device)
        #dim_td = self.temperature ** (3 * (dim_td // 3) / self.num_pos_feats[2])

        pos_x = x_embed[:, :, :, None] / dim_tx
        pos_y = y_embed[:, :, :, None] / dim_ty
        #pos_d = d_embed[:, :, :, :, None] / dim_td
        #print("####pos_x#####",pos_x.shape)
        #print("####pos_y#####",pos_y.shape)


        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        #pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        #print("####pos_x#####",pos_x.shape)
        #print("####pos_y#####",pos_y.shape)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        #print("####pos#####",pos.shape)
        return pos


def build_position_encoding(mode, hidden_dim):
    N_steps = hidden_dim // 2
    if (hidden_dim % 2) != 0:
        N_steps = [N_steps, N_steps + hidden_dim % 3]
    else:
        N_steps = [N_steps, N_steps]
    #print("#######N_steps######",N_steps)

    if mode in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)
    else:
        raise ValueError(f"not supported {mode}")
    #print("#####position_embedding#######",position_embedding.shape)
    return position_embedding
