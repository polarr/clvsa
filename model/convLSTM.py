import torch
import torch.nn as nn
from typing import Tuple

from model.config import ModelConfig

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
        self,
        x: torch.Tensor,
        hidden=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs, (h_n, c_n) = self.lstm(x, hidden)
        return outputs, (h_n, c_n)


class RowSharedConvLSTMCell(nn.Module):
    """
    ConvLSTM cell with kernels moving only horizontally across the 6-column axis
    and shared across feature rows.

    Input:
      x: (B, rows, C_in, cols)

    Hidden/state:
      h, c: (B, rows, C_hidden, cols)
    """
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int,
        rows: int,
        cols: int,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.rows = rows
        self.cols = cols

        padding = kernel_size // 2

        self.conv_x = nn.Conv1d(
            in_channels=input_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_h = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(
            batch_size, self.rows, self.hidden_channels, self.cols,
            device=device, dtype=dtype
        )
        c = torch.zeros(
            batch_size, self.rows, self.hidden_channels, self.cols,
            device=device, dtype=dtype
        )
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        b = x.size(0)

        x_rows = x.reshape(b * self.rows, self.input_channels, self.cols)
        h_rows = h_prev.reshape(b * self.rows, self.hidden_channels, self.cols)

        gates = self.conv_x(x_rows) + self.conv_h(h_rows)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_rows_prev = c_prev.reshape(b * self.rows, self.hidden_channels, self.cols)
        c_rows = f * c_rows_prev + i * g
        h_rows = o * torch.tanh(c_rows)

        h = h_rows.reshape(b, self.rows, self.hidden_channels, self.cols)
        c = c_rows.reshape(b, self.rows, self.hidden_channels, self.cols)
        return h, c