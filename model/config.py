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

    # conv settings
    conv_channels: int = 32
    conv_kernel_size: int = 3
<<<<<<< HEAD

    # kept for CLI compatibility
    conv_proj_dim: int = 128
    use_block_conv: bool = False
    
    #
=======
    use_row_specific_conv: bool = False
>>>>>>> 95b1a481e08db2debbb2fc993127eb89e6a210c9
