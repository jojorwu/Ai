"""
Utilities for setting up tests.
"""
from src.config import LoraConfig, TrainConfig
from src.data.tokenizer import Tokenizer

def create_test_config_and_data(lora_enabled=False):
    """Creates a minimal configuration and dummy data for testing."""
    config = TrainConfig.from_json('config_train.json')
    config.model.d_model = 16
    config.model.num_heads = 2
    config.model.num_kv_heads = 2
    config.model.d_ff = 32
    config.model.num_layers = 1
    config.evolution.pretrain_epochs = 1
    config.evolution.evolution_epochs = 1
    config.evolution.num_agents = 2
    config.evolution.num_survivors = 1

    if not lora_enabled:
        config.lora = None
    elif config.lora is None: # Ensure lora is enabled if requested
        config.lora = LoraConfig()


    tokenizer = Tokenizer('data')
    tokenizer.vocab = {'a': 0, 'b': 1, 'c': 2, '<pad>': 3}
    tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}

    train_data = [0, 1, 2] * 10
    val_data = [2, 1, 0] * 5

    return config, tokenizer, train_data, val_data
