import torch.nn as nn

from model.config import ModelConfig
from model.lstm import *
from model.lsa import *
from model.clsa import *
from model.clvsa import *

def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    model_name = model_name.lower()

    # CLVSA
    if model_name == "clvsa":
        return CLVSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=True,
        )
    
    # CLSA family
    if model_name == "clsa":
        return CLSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=True,
        )

    # Inter-attention only
    if model_name == "clsa_inter":
        return CLSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=True,
        )

    # Self-attention only
    if model_name == "clsa_self":
        return CLSAModel(
            config,
            use_encoder_self_attention=True,
            use_decoder_self_attention=True,
            use_inter_attention=False,
        )
    
    if model_name == "cls":
        return CLSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )
    
    # LSA family
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
    
    if model_name == "ls":
        return LSAModel(
            config,
            use_encoder_self_attention=False,
            use_decoder_self_attention=False,
            use_inter_attention=False,
        )
    
    # Plain LSTM
    if model_name == "lstm":
        return LSTMBaseline(config)

    raise ValueError(f"Unknown model_name: {model_name}")