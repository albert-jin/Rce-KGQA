import torch
import numpy as np
from typing import Optional


def create_src_lengths_mask(batch_size: int, src_lengths: torch.Tensor, max_src_len: Optional[int] = None):
    """
    Generate boolean mask to prevent attention beyond the end of source
    Inputs:
      batch_size : int
      src_lengths : [batch_size] of sentence lengths
      max_src_len: Optionally override max_src_len for the mask
    Outputs:
      [batch_size, max_src_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
    return torch.nn.functional.softmax(scores.float(), dim=-1).type_as(scores)


class Attention_layer(torch.nn.Module):
    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super(Attention_layer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking
        self.proj_w = torch.nn.Linear(self.hidden_dim, self.attention_dim, bias=True)
        self.proj_v = torch.nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, inputs, inputs_length):
        """
            inputs: seq_len * batch_size * hidden_dim
            inputs_length: batch_size
            return: batch_size * seq_len, batch_size * hidden_dim
        """
        seq_len, batch_size, _ = inputs.size()
        flat_inputs = inputs.reshape(-1, self.hidden_dim)
        mlp_inputs = self.proj_w(flat_inputs)
        attention_scores = self.proj_v(mlp_inputs).view(seq_len, batch_size).t()
        normalized_masked_att_scores = masked_softmax(attention_scores, inputs_length, self.src_length_masking).t()
        attention_inputs = (inputs * normalized_masked_att_scores.unsqueeze(2)).sum(0)
        return attention_inputs
