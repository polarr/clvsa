import torch
import torch.nn as nn

from model.config import ModelConfig
from model.convLSTM import RowSharedConvLSTMCell, RowSpecificConvLSTMCell

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