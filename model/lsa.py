from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import DotProductSelfAttention, DotProductInterAttention
from model.common import *

class LSAEncoder(nn.Module):
    """
    LSTM encoder with optional causal self-attention.

    Input:
      x: (B, T_enc, input_dim)

    Output:
      encoder_outputs: (B, T_enc, hidden_dim)
      hidden: list of (h, c), one tuple per LSTM layer
    """
    def __init__(
        self,
        config: ModelConfig,
        use_self_attention: bool = True,
    ):
        super().__init__()

        if config.bidirectional:
            raise ValueError("LSA currently supports bidirectional=False only.")

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.state_dim = config.hidden_dim
        self.use_self_attention = use_self_attention

        self.cells = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            layer_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim

            self.cells.append(
                nn.LSTMCell(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_dim,
                )
            )

            if self.use_self_attention:
                self.self_attn.append(DotProductSelfAttention(self.state_dim))

    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        hidden = []

        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            hidden.append((h, c))

        return hidden

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        b, t, _ = x.shape

        hidden = self._init_hidden(b, x.device, x.dtype)

        raw_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self_attn_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        final_outputs = []

        for step in range(t):
            layer_input = x[:, step, :]
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))

                if self.use_self_attention:
                    h_out, self_weights = self.self_attn[layer_idx].forward_step(
                        h_raw,
                        raw_histories[layer_idx],
                        refined_histories[layer_idx],
                    )
                else:
                    h_out = h_raw
                    self_weights = h_raw.new_zeros(b, step)

                raw_histories[layer_idx].append(h_raw)
                refined_histories[layer_idx].append(h_out)

                if return_attention and self.use_self_attention:
                    padded = h_raw.new_zeros(b, t)
                    if step > 0:
                        padded[:, :step] = self_weights
                    self_attn_maps_per_layer[layer_idx].append(padded.unsqueeze(1))

                new_hidden.append((h_raw, c_new))

                layer_input = h_out

            hidden = new_hidden
            final_outputs.append(refined_histories[-1][-1].unsqueeze(1))

        encoder_outputs = torch.cat(final_outputs, dim=1)

        attention_dict: Dict[str, torch.Tensor] = {}
        if return_attention and self.use_self_attention:
            attention_dict = {
                f"lsa_encoder_self_layer_{i}": torch.cat(self_attn_maps_per_layer[i], dim=1)
                for i in range(self.num_layers)
            }

        return encoder_outputs, hidden, attention_dict


class LSADecoder(nn.Module):
    """
    LSTM decoder with optional inter-attention and optional causal self-attention.

    Per decoder layer:
      LSTMCell raw state
      -> optional inter-attention over encoder outputs
      -> optional decoder self-attention over previous decoder states
      -> next decoder layer / prediction head
    """
    def __init__(
        self,
        config: ModelConfig,
        use_inter_attention: bool = True,
        use_self_attention: bool = True,
    ):
        super().__init__()

        if config.bidirectional:
            raise ValueError("LSA currently supports bidirectional=False only.")

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.state_dim = config.hidden_dim

        self.use_inter_attention = use_inter_attention
        self.use_self_attention = use_self_attention

        self.cells = nn.ModuleList()
        self.inter_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            layer_input_dim = self.input_dim if layer_idx == 0 else self.hidden_dim

            self.cells.append(
                nn.LSTMCell(
                    input_size=layer_input_dim,
                    hidden_size=self.hidden_dim,
                )
            )

            if self.use_inter_attention:
                self.inter_attn.append(DotProductInterAttention(self.state_dim))

            if self.use_self_attention:
                self.self_attn.append(DotProductSelfAttention(self.state_dim))

        self.head = PredictionHead(
            input_dim=self.hidden_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_outputs: torch.Tensor,
        init_hidden: List[Tuple[torch.Tensor, torch.Tensor]],
        return_attention: bool = False,
    ):
        b, t, _ = x.shape

        hidden = [(h.clone(), c.clone()) for (h, c) in init_hidden]

        pre_self_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        inter_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        dec_self_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        logits_steps = []

        for step in range(t):
            layer_input = x[:, step, :]
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))

                h_vec = h_raw

                if self.use_inter_attention:
                    h_vec, inter_weights = self.inter_attn[layer_idx](
                        h_vec,
                        encoder_outputs,
                    )

                    if return_attention:
                        inter_maps_per_layer[layer_idx].append(inter_weights.unsqueeze(1))

                if self.use_self_attention:
                    h_pre_self = h_vec

                    h_vec, self_weights = self.self_attn[layer_idx].forward_step(
                        h_pre_self,
                        pre_self_histories[layer_idx],
                        refined_histories[layer_idx],
                    )

                    pre_self_histories[layer_idx].append(h_pre_self)

                    if return_attention:
                        padded_self = h_vec.new_zeros(b, t)
                        if step > 0:
                            padded_self[:, :step] = self_weights
                        dec_self_maps_per_layer[layer_idx].append(padded_self.unsqueeze(1))

                refined_histories[layer_idx].append(h_vec)

                new_hidden.append((h_raw, c_new))

                layer_input = h_vec

            hidden = new_hidden

            top_vec = refined_histories[-1][-1]
            logits_steps.append(self.head(top_vec).unsqueeze(1))

        logits_seq = torch.cat(logits_steps, dim=1)

        if not return_attention:
            return logits_seq

        attention_dict: Dict[str, torch.Tensor] = {}

        if self.use_inter_attention:
            for i in range(self.num_layers):
                attention_dict[f"lsa_inter_layer_{i}"] = torch.cat(
                    inter_maps_per_layer[i],
                    dim=1,
                )

        if self.use_self_attention:
            for i in range(self.num_layers):
                attention_dict[f"lsa_decoder_self_layer_{i}"] = torch.cat(
                    dec_self_maps_per_layer[i],
                    dim=1,
                )

        return logits_seq, attention_dict


class LSAModel(nn.Module):
    """
    LSTM Seq2Seq with toggleable attention.

    Current 30-minute input:
      x_enc: (B, 13, 30)
      x_dec: (B, 13, 30)

    Future 5-minute input also works if the dataset emits:
      x_enc: (B, T, 5)
      x_dec: (B, T, 5)
      config.input_dim = 5
    """
    def __init__(
        self,
        config: ModelConfig,
        use_encoder_self_attention: bool = True,
        use_decoder_self_attention: bool = True,
        use_inter_attention: bool = True,
    ):
        super().__init__()

        self.encoder = LSAEncoder(
            config,
            use_self_attention=use_encoder_self_attention,
        )

        self.decoder = LSADecoder(
            config,
            use_inter_attention=use_inter_attention,
            use_self_attention=use_decoder_self_attention,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        return_attention: bool = False,
    ):
        encoder_outputs, encoder_hidden, enc_attn = self.encoder(
            x_enc,
            return_attention=return_attention,
        )

        out = self.decoder(
            x_dec,
            encoder_outputs=encoder_outputs,
            init_hidden=encoder_hidden,
            return_attention=return_attention,
        )

        if not return_attention:
            return out

        logits_seq, dec_attn = out
        attn = {}
        attn.update(enc_attn)
        attn.update(dec_attn)
        return logits_seq, attn