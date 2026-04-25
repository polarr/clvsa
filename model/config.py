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
    use_row_specific_conv: bool = False

    # variational settings
    latent_dim: int = 32 # unspecified in paper
    prior_fc_dim: int = 512
    posterior_fc_dim: int = 256
    alpha: float = 2.5e-4
    beta_max: float = 1.0