import torch
import torch.nn as nn

from model.config import ModelConfig
from model.convLSTM import LSTMEncoder
from model.common import *

class LSTMBaseline(nn.Module):
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