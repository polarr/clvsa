from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.convLSTM import LSTMEncoder, RowSharedConvLSTMCell
from model.attention import DotProductSelfAttention, DotProductInterAttention

class PredictionHead(nn.Module):
    """
    Paper classifier head:
      200-unit FC -> 50-unit FC -> softmax logits

    Dropout only on FC blocks.
    """
    def __init__(self, input_dim: int, dropout: float, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DayLSTMTagger(nn.Module):
    """
    Plain LSTM baseline on the same day-paired supervision.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        seq_config = ModelConfig(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            output_dim=config.output_dim,
            block_rows=config.block_rows,
            block_cols=config.block_cols,
            conv_channels=config.conv_channels,
            conv_kernel_size=config.conv_kernel_size,
        )
        self.encoder = LSTMEncoder(seq_config)

        encoder_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.head = PredictionHead(
            input_dim=encoder_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def forward(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_enc, x_dec], dim=1)            # (B, 26, 30)
        outputs, _ = self.encoder(x)
        dec_outputs = outputs[:, x_enc.size(1):, :]     # (B, 13, H)
        logits_seq = self.head(dec_outputs)             # (B, 13, C)
        return logits_seq


class CLSAEncoder(nn.Module):
    """
    2-layer ConvLSTM encoder with self-attention in each layer.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rows = config.block_rows
        self.cols = config.block_cols
        self.channels = config.conv_channels
        self.num_layers = config.num_layers
        self.state_dim = self.rows * self.cols * self.channels

        expected_dim = self.rows * self.cols
        if config.input_dim != expected_dim:
            raise ValueError(f"Expected input_dim={expected_dim}, got {config.input_dim}")

        self.cells = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 1 if layer_idx == 0 else self.channels
            self.cells.append(
                RowSharedConvLSTMCell(
                    input_channels=in_channels,
                    hidden_channels=self.channels,
                    kernel_size=config.conv_kernel_size,
                    rows=self.rows,
                    cols=self.cols,
                )
            )
            self.self_attn.append(DotProductSelfAttention(self.state_dim))

    def _prepare_frames(self, x_flat: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_flat.shape
        x = x_flat.contiguous().view(b, t, self.rows, self.cols)  # (B, T, 5, 6)
        x = x.unsqueeze(3)                                         # (B, T, 5, 1, 6)
        return x

    def _vec_to_frame(self, x_vec: torch.Tensor) -> torch.Tensor:
        return x_vec.view(x_vec.size(0), self.rows, self.channels, self.cols)

    def _frame_to_vec(self, x_frame: torch.Tensor) -> torch.Tensor:
        return x_frame.reshape(x_frame.size(0), -1)

    def forward(
        self,
        x_flat: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        x = self._prepare_frames(x_flat)
        b, t, _, _, _ = x.shape

        hidden = [cell.init_state(b, x.device, x.dtype) for cell in self.cells]

        raw_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        self_attn_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        final_outputs = []

        for step in range(t):
            layer_input = x[:, step]  # (B, rows, 1, cols)
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))

                h_raw_vec = self._frame_to_vec(h_raw)
                h_refined_vec, self_weights = self.self_attn[layer_idx].forward_step(
                    h_raw_vec,
                    raw_histories[layer_idx],
                    refined_histories[layer_idx],
                )

                raw_histories[layer_idx].append(h_raw_vec)
                refined_histories[layer_idx].append(h_refined_vec)

                padded = h_raw_vec.new_zeros(b, t)
                if step > 0:
                    padded[:, :step] = self_weights
                self_attn_maps_per_layer[layer_idx].append(padded.unsqueeze(1))

                h_refined = self._vec_to_frame(h_refined_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_refined

            hidden = new_hidden
            final_outputs.append(refined_histories[-1][-1].unsqueeze(1))

        encoder_outputs = torch.cat(final_outputs, dim=1)  # (B, T, D)

        attention_dict = {
            f"encoder_self_layer_{i}": torch.cat(self_attn_maps_per_layer[i], dim=1)
            for i in range(self.num_layers)
        }
        return encoder_outputs, hidden, attention_dict


class CLSADecoder(nn.Module):
    """
    2-layer ConvLSTM decoder with:
    - inter-attention from final encoder layer to both decoder layers
    - self-attention in each decoder layer
    - requested order: ConvLSTM -> inter-attn -> self-attn
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.rows = config.block_rows
        self.cols = config.block_cols
        self.channels = config.conv_channels
        self.num_layers = config.num_layers
        self.state_dim = self.rows * self.cols * self.channels

        self.cells = nn.ModuleList()
        self.inter_attn = nn.ModuleList()
        self.self_attn = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            in_channels = 1 if layer_idx == 0 else self.channels
            self.cells.append(
                RowSharedConvLSTMCell(
                    input_channels=in_channels,
                    hidden_channels=self.channels,
                    kernel_size=config.conv_kernel_size,
                    rows=self.rows,
                    cols=self.cols,
                )
            )
            self.inter_attn.append(DotProductInterAttention(self.state_dim))
            self.self_attn.append(DotProductSelfAttention(self.state_dim))

        self.head = PredictionHead(
            input_dim=self.state_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def _prepare_frames(self, x_flat: torch.Tensor) -> torch.Tensor:
        b, t, _ = x_flat.shape
        x = x_flat.contiguous().view(b, t, self.rows, self.cols)  # (B, T, 5, 6)
        x = x.unsqueeze(3)                                         # (B, T, 5, 1, 6)
        return x

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

        # For requested order, self-attn sees previous inter-attended states as h_i.
        pre_self_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        refined_histories: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        inter_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]
        dec_self_maps_per_layer: List[List[torch.Tensor]] = [[] for _ in range(self.num_layers)]

        logits_steps = []

        for step in range(t):
            layer_input = x[:, step]  # (B, rows, 1, cols)
            new_hidden = []

            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = hidden[layer_idx]
                h_raw, c_new = cell(layer_input, (h_prev, c_prev))
                h_raw_vec = self._frame_to_vec(h_raw)

                # inter-attention first
                h_inter_vec, inter_weights = self.inter_attn[layer_idx](
                    h_raw_vec,
                    encoder_outputs,
                )

                # then self-attention
                h_refined_vec, self_weights = self.self_attn[layer_idx].forward_step(
                    h_inter_vec,
                    pre_self_histories[layer_idx],
                    refined_histories[layer_idx],
                )

                pre_self_histories[layer_idx].append(h_inter_vec)
                refined_histories[layer_idx].append(h_refined_vec)

                padded_inter = h_raw_vec.new_zeros(b, encoder_outputs.size(1))
                padded_inter[:, :encoder_outputs.size(1)] = inter_weights
                inter_maps_per_layer[layer_idx].append(padded_inter.unsqueeze(1))

                padded_self = h_raw_vec.new_zeros(b, t)
                if step > 0:
                    padded_self[:, :step] = self_weights
                dec_self_maps_per_layer[layer_idx].append(padded_self.unsqueeze(1))

                h_refined = self._vec_to_frame(h_refined_vec)
                new_hidden.append((h_raw, c_new))
                layer_input = h_refined

            hidden = new_hidden
            top_vec = refined_histories[-1][-1]
            logits_steps.append(self.head(top_vec).unsqueeze(1))

        logits_seq = torch.cat(logits_steps, dim=1)  # (B, T, C)

        if not return_attention:
            return logits_seq

        attention_dict = {}
        for i in range(self.num_layers):
            attention_dict[f"inter_layer_{i}"] = torch.cat(inter_maps_per_layer[i], dim=1)
            attention_dict[f"decoder_self_layer_{i}"] = torch.cat(dec_self_maps_per_layer[i], dim=1)

        return logits_seq, attention_dict


class CLSAModel(nn.Module):
    """
    CLSA forward model:
    - 2-layer ConvLSTM encoder
    - self-attention in each encoder layer
    - 2-layer ConvLSTM decoder
    - inter-attention into both decoder layers
    - self-attention in each decoder layer
    - requested decoder order: inter-attn before self-attn
    - classifier head: 200 -> 50 -> softmax logits
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = CLSAEncoder(config)
        self.decoder = CLSADecoder(config)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        return_attention: bool = False,
    ):
        encoder_outputs, encoder_hidden, enc_attn = self.encoder(x_enc)

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


def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lstm":
        return DayLSTMTagger(config)
    if model_name == "clsa":
        return CLSAModel(config)

    raise ValueError(f"Unknown model_name: {model_name}")