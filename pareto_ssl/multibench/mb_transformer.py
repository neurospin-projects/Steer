"""Sequence encoder for the MultiBench affect datasets.

The architecture is the one FactorCL uses for MultiBench timeseries: a 1x1 Conv1d
lifting raw per-frame features to the model width, then a 5-layer pre-norm
TransformerEncoder with 5 heads, with the last token taken as the embedding.

It is written out here so the benchmark does not need a third-party checkout to
build a model. Parameter names and shapes match the reference implementation, so
checkpoints trained before this module existed still load unchanged -
`tests_transformer_equivalence.py` asserts exactly that.
"""
import math

import torch
import torch.nn as nn
from einops import rearrange


def sincos_posemb(max_seq_len: int, embed_dim: int = 1024) -> torch.Tensor:
    """Fixed sin/cos table, shape (1, max_seq_len, embed_dim)."""
    if embed_dim % 2 != 0:
        raise ValueError(f"sin/cos positional encoding needs an even dim, got {embed_dim}")
    pe = torch.zeros(max_seq_len, embed_dim)
    pos = torch.arange(0, max_seq_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float)
                    * -(math.log(10000.0) / embed_dim))
    pe[:, 0::2] = torch.sin(pos.float() * div)
    pe[:, 1::2] = torch.cos(pos.float() * div)
    return pe[None, :, :]


class Transformer(nn.Module):
    """Per-frame features -> token sequence -> pre-norm TransformerEncoder."""

    def __init__(self, n_features: int, dim: int, max_seq_length: int = 50,
                 return_seq: bool = True, positional_encoding: bool = True,
                 pad_value=None):
        super().__init__()
        self.embed_dim = dim
        self.return_seq = return_seq
        self.use_positional_embedding = positional_encoding
        self.pad_value = pad_value
        self.conv = nn.Conv1d(n_features, dim, kernel_size=1, padding=0, bias=False)
        self.positional_embedding = nn.Parameter(
            sincos_posemb(max_seq_length, dim), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=5, batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)

    def forward(self, x):
        x = self.conv(rearrange(x, "b l n -> b n l"))
        x = rearrange(x, "b n l -> b l n")
        if self.use_positional_embedding:
            x = x + self.positional_embedding[:, :x.size(1)]
        x = self.transformer(x)
        if not self.return_seq:
            x = x[:, -1]
        return x
