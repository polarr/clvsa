import torch
import torch.nn as nn
import math

from typing import Tuple, List

class DotProductSelfAttention(nn.Module):
    """
    Paper-style causal self-attention:

      a_t = softmax(h_t^T H'_{1:t-1})
      c_t = sum_i a_ti h_i
      h'_t = tanh(W_sa [h_t, c_t])

    Score against previous refined states H', but aggregate previous states h.
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.fuse = nn.Linear(2 * state_dim, state_dim)

    def forward_step(
        self,
        state_t: torch.Tensor,                  # current state before self-attn, (B, D)
        prev_states: List[torch.Tensor],        # previous pre-self states h_i
        prev_refined_states: List[torch.Tensor] # previous refined states h'_i
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(prev_states) == 0:
            context = torch.zeros_like(state_t)
            attn_weights = state_t.new_zeros(state_t.size(0), 0)
        else:
            prev = torch.stack(prev_states, dim=1)                  # (B, t-1, D)
            prev_refined = torch.stack(prev_refined_states, dim=1)  # (B, t-1, D)

            scores = torch.bmm(prev_refined, state_t.unsqueeze(-1)).squeeze(-1)
            scores = scores / math.sqrt(state_t.size(-1))
            attn_weights = torch.softmax(scores, dim=1)
            context = torch.sum(prev_refined * attn_weights.unsqueeze(-1), dim=1)

        refined_state = torch.tanh(self.fuse(torch.cat([state_t, context], dim=-1)))
        return refined_state, attn_weights

class DotProductInterAttention(nn.Module):
    """
    Paper-style inter-attention:

      a_td = softmax(h_d_t^T H_e)
      c_td = sum_e a_tde h_e
      h'_d_t = tanh(W_ia [h_d_t, c_td])
    """
    def __init__(self, state_dim: int):
        super().__init__()
        self.fuse = nn.Linear(2 * state_dim, state_dim)

    def forward(
        self,
        decoder_state: torch.Tensor,    # (B, D)
        encoder_outputs: torch.Tensor,  # (B, T_enc, D)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = torch.bmm(encoder_outputs, decoder_state.unsqueeze(-1)).squeeze(-1)
        scores = scores / math.sqrt(decoder_state.size(-1))
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(encoder_outputs * attn_weights.unsqueeze(-1), dim=1)
        refined_state = torch.tanh(self.fuse(torch.cat([decoder_state, context], dim=-1)))
        return refined_state, attn_weights