from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from model.config import ModelConfig
from model.convLSTM import LSTMEncoder, RowSharedConvLSTMCell, RowSpecificConvLSTMCell
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

def build_convlstm_cell(
    config: ModelConfig,
    input_channels: int,
) -> nn.Module:
    cell_cls = RowSpecificConvLSTMCell if config.use_row_specific_conv else RowSharedConvLSTMCell

    return cell_cls(
        input_channels=input_channels,
        hidden_channels=config.conv_channels,
        kernel_size=config.conv_kernel_size,
        rows=config.block_rows,
        cols=config.block_cols,
    )

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


# Backward-compatible alias: pure ConvLSTM seq2seq.
class CLS(CLSAModel):
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )

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


class LS(LSAModel):
    """
    Pure LSTM encoder-decoder baseline:
      no encoder self-attention
      no decoder self-attention
      no inter-attention
    """
    def __init__(self, config: ModelConfig):
        super().__init__(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )

def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lstm":
        return DayLSTMTagger(config)

    if model_name == "cls":
        return CLSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )

    if model_name == "clsa":
        return CLSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=True,
        )

    # Inter-attention only: no encoder/decoder self-attention.
    if model_name == "clsa_inter":
        return CLSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=True,
        )

    # Self-attention only: encoder + decoder self-attention, no inter-attention.
    if model_name == "clsa_self":
        return CLSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=False,
        )

    # LSTM seq2seq + attention family.
    if model_name == "ls":
        return LSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )

    if model_name == "lsa":
        return LSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=True,
        )

    if model_name == "lsa_inter":
        return LSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=True,
        )

    if model_name == "lsa_self":
        return LSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=False,
        )

    raise ValueError(f"Unknown model_name: {model_name}")