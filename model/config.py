from dataclasses import dataclass

@dataclass
class ModelConfig:
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    output_dim: int = 3

    # rows = features, cols = 6 consecutive 5-minute steps
    block_rows: int = 5
    block_cols: int = 6

    # paper settings
    conv_channels: int = 32
    conv_kernel_size: int = 3

    # kept for CLI compatibility
    conv_proj_dim: int = 128
    use_block_conv: bool = False
    
    #