from collections import OrderedDict
from typing import Tuple, Union
import numpy as np, torch
import torch.nn.functional as F
from torch import nn



class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ('c_fc', nn.Linear(d_model, d_model * 4)),
            ('gelu', QuickGELU()), ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=(x.dtype), device=(x.device)) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=(self.attn_mask))[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Encoder(nn.Module):

    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = (nn.Sequential)(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TextTransformer(nn.Module):
    #768 77 49408 768 12 12
    def __init__(self, embed_dim = 512,
                 context_length = 77,
                 vocab_size = 49408,
                 transformer_width = 768,
                 transformer_heads = 8,
                 transformer_layers = 10):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer = Encoder(width=transformer_width,
                                       layers=transformer_layers,
                                       heads=transformer_heads,
                                       attn_mask=(self.build_attention_mask()))

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim).float()).float()
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = self.transformer.width ** (-0.5) * (2 * self.transformer.layers) ** (-0.5)
        attn_std = self.transformer.width ** (-0.5)
        fc_std = (2 * self.transformer.width) ** (-0.5)
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_((self.text_projection), std=(self.transformer.width ** (-0.5)))

    def forward(self, text, norm=False, is_vis=False):
        if not is_vis:
            x = self.token_embedding(text)
            x = x + self.positional_embedding
        else:
            x = text
            x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        if is_vis:
            text_features = x[(torch.arange(x.shape[0]),  (torch.ones(x.shape[0])*10).long()) ] @ self.text_projection
        else:
            text_features = x[(torch.arange(x.shape[0]), text.argmax(dim=(-1)))] @ self.text_projection
        if norm:
            text_features = text_features / text_features.norm(dim=(-1), keepdim=True)
        return text_features

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)
        return mask