"""
This module contains utility functions for tests.
"""
from config import Config, TransformerConfig
from model import Transformer


def create_test_model(vocab_size=50, ltm=True):
    """Creates a Transformer model for testing."""
    config = Config.from_json('config.json')
    if ltm:
        config.model.ltm_d_hidden = 64
        config.model.ltm_num_layers = 2
    else:
        config.model.ltm_d_hidden = None
        config.model.ltm_num_layers = None

    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm
    )
    model = Transformer(transformer_config)
    return model, config
