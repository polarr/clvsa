from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.attention import DotProductSelfAttention, DotProductInterAttention
from model.common import *

class CLSAEncoder(nn.Module):
    """
    Toggleable ConvLSTM encoder.

    If use_self_attention=False:
      ConvLSTM only; output at each step is raw top-layer ConvLSTM state.

    If use_self_attention=True:
      ConvLSTM -> causal self-attention at each layer; output at each step is
      top-layer refined state.
    """
    def __init__(
        self,
        config: ModelConfig,
        use_self_attention: bool = True,
    ):
        super().__init__()
        self.rows = config.block_rows
        self.cols = config.block_cols
        self.channels = config.conv_channels
        self.num_layers = config.num_layers
        self.state_dim = self.rows * self.cols * self.channels
        self.use_self_attention = use_self_attention

        expected_dim = self.rows * self.cols
        if config.input_dim != expected_dim:
            raise ValueError(f"Expected input_dim={expected_dim}, got {config.input_dim}")

        self.cells = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 1 if layer_idx == 0 else self.channels
            self.cells.append(
                build_convlstm_cell(config, input_channels=in_channels)
            )
            if self.use_self_attention:
                self.self_attn.append(DotProductSelfAttention(self.state_dim))

    def _prepare_frames(self, x_flat: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_flat.shape
        x = x_flat.contiguous().view(b, t, self.rows, self.cols)  # (B, T, rows, cols)
        return x.unsqueeze(3)                                      # (B, T, rows, 1, cols)

    def _vec_to_frame(self, x_vec: torch.Tensor) -> torch.Tensor:
        return x_vec.view(x_vec.size(0), self.rows, self.channels, self.cols)

    def _frame_to_vec(self, x_frame: torch.Tensor) -> torch.Tensor:
        return x_frame.reshape(x_frame.size(0), -1)

    def forward(
        self,
        x_flat: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        x = self._prepare_frames(x_flat)
        b, t, _, _, _ = x.shape

        hidden = [cell.init_state(b, x.device, x.dtype) for cell in self.cells]

        raw_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self_attn_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        final_outputs = []

        for step in range(t):
            layer_input = x[:, step]  # (B, rows, C_in, cols)
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))
                h_raw_vec = self._frame_to_vec(h_raw)

                if self.use_self_attention:
                    h_out_vec, self_weights = self.self_attn[layer_idx].forward_step(
                        h_raw_vec,
                        raw_histories[layer_idx],
                        refined_histories[layer_idx],
                    )
                else:
                    h_out_vec = h_raw_vec
                    self_weights = h_raw_vec.new_zeros(b, step)

                raw_histories[layer_idx].append(h_raw_vec)
                refined_histories[layer_idx].append(h_out_vec)

                if return_attention and self.use_self_attention:
                    padded = h_raw_vec.new_zeros(b, t)
                    if step > 0:
                        padded[:, :step] = self_weights
                    self_attn_maps_per_layer[layer_idx].append(padded.unsqueeze(1))

                h_out = self._vec_to_frame(h_out_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_out

            hidden = new_hidden
            final_outputs.append(refined_histories[-1][-1].unsqueeze(1))

        encoder_outputs = torch.cat(final_outputs, dim=1)  # (B, T, D)

        attention_dict: Dict[str, torch.Tensor] = {}
        if return_attention and self.use_self_attention:
            attention_dict = {
                f"encoder_self_layer_{i}": torch.cat(self_attn_maps_per_layer[i], dim=1)
                for i in range(self.num_layers)
            }

        return encoder_outputs, hidden, attention_dict


class CLSADecoder(nn.Module):
    """
    Toggleable ConvLSTM decoder.

    Per layer, per decoder step:
      ConvLSTM raw state
      -> optional inter-attention over encoder outputs
      -> optional decoder self-attention over previous decoder states
      -> pass to next decoder layer / prediction head
    """
    def __init__(
        self,
        config: ModelConfig,
        use_inter_attention: bool = True,
        use_self_attention: bool = True,
    ):
        super().__init__()
        self.rows = config.block_rows
        self.cols = config.block_cols
        self.channels = config.conv_channels
        self.num_layers = config.num_layers
        self.state_dim = self.rows * self.cols * self.channels

        self.use_inter_attention = use_inter_attention
        self.use_self_attention = use_self_attention

        self.cells = nn.ModuleList()
        self.inter_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 1 if layer_idx == 0 else self.channels
            self.cells.append(
                build_convlstm_cell(config, input_channels=in_channels)
            )
            if self.use_inter_attention:
                self.inter_attn.append(DotProductInterAttention(self.state_dim))
            if self.use_self_attention:
                self.self_attn.append(DotProductSelfAttention(self.state_dim))

        self.head = PredictionHead(
            input_dim=self.state_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def _prepare_frames(self, x_flat: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_flat.shape
        x = x_flat.contiguous().view(b, t, self.rows, self.cols)  # (B, T, rows, cols)
        return x.unsqueeze(3)                                      # (B, T, rows, 1, cols)

    def _vec_to_frame(self, x_vec: torch.Tensor) -> torch.Tensor:
        return x_vec.view(x_vec.size(0), self.rows, self.channels, self.cols)

    def _frame_to_vec(self, x_frame: torch.Tensor) -> torch.Tensor:
        return x_frame.reshape(x_frame.size(0), -1)

    def forward(
        self,
        x_flat: torch.Tensor,
        encoder_outputs: torch.Tensor,
        init_hidden: List[Tuple[torch.Tensor, torch.Tensor]],
        return_attention: bool = False,
    ):
        x = self._prepare_frames(x_flat)
        b, t, _, _, _ = x.shape

        hidden = [(h.clone(), c.clone()) for (h, c) in init_hidden]

        # self-attention memory is separate per decoder layer
        pre_self_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        inter_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        dec_self_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        logits_steps = []

        for step in range(t):
            layer_input = x[:, step]  # (B, rows, C_in, cols)
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))
                h_vec = self._frame_to_vec(h_raw)

                if self.use_inter_attention:
                    h_vec, inter_weights = self.inter_attn[layer_idx](
                        h_vec,
                        encoder_outputs,
                    )
                    if return_attention:
                        inter_maps_per_layer[layer_idx].append(inter_weights.unsqueeze(1))

                if self.use_self_attention:
                    h_pre_self_vec = h_vec
                    h_vec, self_weights = self.self_attn[layer_idx].forward_step(
                        h_pre_self_vec,
                        pre_self_histories[layer_idx],
                        refined_histories[layer_idx],
                    )
                    pre_self_histories[layer_idx].append(h_pre_self_vec)

                    if return_attention:
                        padded_self = h_vec.new_zeros(b, t)
                        if step > 0:
                            padded_self[:, :step] = self_weights
                        dec_self_maps_per_layer[layer_idx].append(padded_self.unsqueeze(1))

                refined_histories[layer_idx].append(h_vec)

                h_out = self._vec_to_frame(h_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_out

            hidden = new_hidden
            top_vec = refined_histories[-1][-1]
            logits_steps.append(self.head(top_vec).unsqueeze(1))

        logits_seq = torch.cat(logits_steps, dim=1)  # (B, T, C)

        if not return_attention:
            return logits_seq

        attention_dict: Dict[str, torch.Tensor] = {}

        if self.use_inter_attention:
            for i in range(self.num_layers):
                attention_dict[f"inter_layer_{i}"] = torch.cat(inter_maps_per_layer[i], dim=1)

        if self.use_self_attention:
            for i in range(self.num_layers):
                attention_dict[f"decoder_self_layer_{i}"] = torch.cat(dec_self_maps_per_layer[i], dim=1)

        return logits_seq, attention_dict


class CLSAModel(nn.Module):
    """
    Unified CLS/CLSA model with toggleable attention.

    Flags:
      use_encoder_self_attention=False,
      use_decoder_self_attention=False,
      use_inter_attention=False
        -> pure ConvLSTM seq2seq baseline, equivalent to CLS.

      all True
        -> full CLSA.
    """
    def __init__(
        self,
        config: ModelConfig,
        use_encoder_self_attention: bool = True,
        use_decoder_self_attention: bool = True,
        use_inter_attention: bool = True
    ):
        super().__init__()
        self.encoder = CLSAEncoder(
            config,
            use_self_attention=use_encoder_self_attention,
        )
        self.decoder = CLSADecoder(
            config,
            use_inter_attention=use_inter_attention,
            use_self_attention=use_decoder_self_attention
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