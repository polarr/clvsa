from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    output_dim: int = 3  # 3-class classification: down / flat / up

    # Block structure: 5 feature rows x 6 time columns
    block_rows: int = 5
    block_cols: int = 6

    # Paper-inspired convolution settings
    conv_channels: int = 32
    conv_kernel_size: int = 3
    conv_proj_dim: int = 128

    # Seq2Seq settings
    use_block_conv: bool = False
    decoder_input_mode: str = "teacher_forcing_features"  # placeholder for future extension


class LSTMEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

    def forward(
        self, x: torch.Tensor, hidden=None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (B, T, F)
        outputs: (B, T, H * D)
        h_n: (L * D, B, H)
        c_n: (L * D, B, H)
        """
        outputs, (h_n, c_n) = self.lstm(x, hidden)
        return outputs, (h_n, c_n)


class SharedHorizontalConvBlock(nn.Module):
    """
    Paper-inspired block encoder for one flattened 30-minute frame.

    Input block:
      flattened 30-dim vector corresponding to a 5x6 frame
      rows = feature types (O, H, L, C, V)
      cols = 6 consecutive 5-minute time steps

    Convolution rule:
      - kernels move only horizontally across columns (time axis)
      - kernels are shared across different feature rows
      - no compression of frame size
      - if output channels = 32, output block shape becomes 5 x 6 x 32
    """
    def __init__(self, config: ModelConfig):
        super().__init__()

        expected_dim = config.block_rows * config.block_cols
        if config.input_dim != expected_dim:
            raise ValueError(
                f"Expected input_dim={expected_dim} from block_rows*block_cols, "
                f"got {config.input_dim}"
            )

        self.block_rows = config.block_rows
        self.block_cols = config.block_cols
        self.conv_channels = config.conv_channels

        self.time_conv = nn.Conv1d(
            in_channels=1,
            out_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            padding=config.conv_kernel_size // 2,
        )

        flattened_conv_dim = self.block_rows * self.block_cols * self.conv_channels
        self.proj = nn.Sequential(
            nn.Linear(flattened_conv_dim, config.conv_proj_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 30)

        returns:
            block_embeddings: (B, T, conv_proj_dim)
        """
        b, t, f = x.shape

        x = x.view(b * t, self.block_rows, self.block_cols)          # (B*T, 5, 6)
        x = x.reshape(b * t * self.block_rows, 1, self.block_cols)   # (B*T*5, 1, 6)

        x = self.time_conv(x)                                         # (B*T*5, C, 6)
        x = torch.relu(x)

        x = x.view(b * t, self.block_rows, self.conv_channels, self.block_cols)
        x = x.permute(0, 1, 3, 2).contiguous()                        # (B*T, 5, 6, C)
        x = x.view(b * t, -1)                                         # (B*T, 5*6*C)

        x = self.proj(x)                                              # (B*T, conv_proj_dim)
        x = x.view(b, t, -1)                                           # (B, T, conv_proj_dim)
        return x


class BahdanauInterAttention(nn.Module):
    """
    Decoder-to-encoder inter-attention.
    At each decoder step, attend over all encoder hidden states.
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attn_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        decoder_state:   (B, H_dec)
        encoder_outputs: (B, T_enc, H_enc)

        returns:
            context:      (B, H_enc)
            attn_weights: (B, T_enc)
        """
        enc_proj = self.encoder_proj(encoder_outputs)                  # (B, T_enc, A)
        dec_proj = self.decoder_proj(decoder_state).unsqueeze(1)       # (B, 1, A)

        energy = torch.tanh(enc_proj + dec_proj)                       # (B, T_enc, A)
        scores = self.v(energy).squeeze(-1)                            # (B, T_enc)
        attn_weights = torch.softmax(scores, dim=1)                    # (B, T_enc)

        context = torch.sum(encoder_outputs * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class PredictionHead(nn.Module):
    def __init__(self, input_dim: int, dropout: float, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = LSTMEncoder(config)

        encoder_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.head = PredictionHead(
            input_dim=encoder_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def _get_final_representation(self, h_n: torch.Tensor) -> torch.Tensor:
        if not self.config.bidirectional:
            final_repr = h_n[-1]
        else:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            final_repr = torch.cat([forward_last, backward_last], dim=-1)
        return final_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, (h_n, c_n) = self.encoder(x)
        final_repr = self._get_final_representation(h_n)
        logits = self.head(final_repr)
        return logits


class Seq2SeqInterAttn(nn.Module):
    """
    Minimal Seq2Seq + inter-attention model.

    - Split input sequence into encoder half and decoder half
    - Encode first half
    - Decode second half step by step
    - At each decoder step, attend over all encoder outputs
    - Use final attended decoder representation for classification
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.use_block_conv:
            self.block_encoder = SharedHorizontalConvBlock(config)
            seq_input_dim = config.conv_proj_dim
        else:
            self.block_encoder = None
            seq_input_dim = config.input_dim

        seq_config = ModelConfig(
            input_dim=seq_input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            bidirectional=False,   # decoder kept unidirectional
            output_dim=config.output_dim,
            block_rows=config.block_rows,
            block_cols=config.block_cols,
            conv_channels=config.conv_channels,
            conv_kernel_size=config.conv_kernel_size,
            conv_proj_dim=config.conv_proj_dim,
            use_block_conv=config.use_block_conv,
        )

        self.encoder = LSTMEncoder(seq_config)
        self.decoder = LSTMEncoder(seq_config)

        encoder_dim = config.hidden_dim
        decoder_dim = config.hidden_dim
        self.inter_attn = BahdanauInterAttention(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            attn_dim=config.hidden_dim,
        )

        self.attn_fusion = nn.Sequential(
            nn.Linear(encoder_dim + decoder_dim, decoder_dim),
            nn.Tanh(),
        )

        self.head = PredictionHead(
            input_dim=decoder_dim,
            dropout=config.dropout,
            output_dim=config.output_dim,
        )

    def _maybe_encode_blocks(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_encoder is not None:
            return self.block_encoder(x)
        return x

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x: (B, T, F)

        Split:
          encoder input = first half
          decoder input = second half

        returns:
          logits: (B, C)
          optionally attention maps from decoder steps
        """
        x = self._maybe_encode_blocks(x)  # (B, T, F_seq)

        b, t, f = x.shape
        if t < 2:
            raise ValueError("Seq2SeqInterAttn requires sequence length >= 2")

        t_enc = t // 2
        t_dec = t - t_enc

        x_enc = x[:, :t_enc, :]   # (B, T_enc, F)
        x_dec = x[:, t_enc:, :]   # (B, T_dec, F)

        encoder_outputs, (h_enc, c_enc) = self.encoder(x_enc)

        decoder_hidden = (h_enc, c_enc)
        attended_states = []
        all_attn_weights = []

        for step in range(t_dec):
            dec_input_step = x_dec[:, step:step + 1, :]                 # (B, 1, F)
            dec_outputs, decoder_hidden = self.decoder(dec_input_step, decoder_hidden)
            dec_state = dec_outputs[:, -1, :]                           # (B, H)

            context, attn_weights = self.inter_attn(dec_state, encoder_outputs)
            fused = self.attn_fusion(torch.cat([dec_state, context], dim=-1))  # (B, H)

            attended_states.append(fused.unsqueeze(1))
            all_attn_weights.append(attn_weights.unsqueeze(1))

        attended_states = torch.cat(attended_states, dim=1)             # (B, T_dec, H)
        all_attn_weights = torch.cat(all_attn_weights, dim=1)           # (B, T_dec, T_enc)

        final_repr = attended_states[:, -1, :]                          # classify final decoder step
        logits = self.head(final_repr)

        if return_attention:
            return logits, all_attn_weights
        return logits


def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "lstm":
        return BaselineLSTM(config)
    if model_name in {"seq2seq_attn", "s2s_attn", "seq2seq_attention"}:
        return Seq2SeqInterAttn(config)

    raise ValueError(f"Unknown model_name: {model_name}")