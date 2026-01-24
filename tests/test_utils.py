"""
Shared utilities for the test suite.
"""
from src.config.core import TrainConfig


class MockTokenizer:  # pylint: disable=too-few-public-methods
    """A mock tokenizer for testing purposes."""
    vocab_size = 50

    def __init__(self):
        self.char_to_idx = {
            '<ASK_FOR_HELP>': 48,
            '<I_DONT_KNOW>': 49
        }

    def encode(self, text):
        """Bare-bones encode method for testing."""
        return list(text.encode('utf-8'))


def create_test_config_and_data():
    """Creates a minimal configuration and dummy data for testing."""
    config = TrainConfig.from_json('config_train.json')
    # Use a smaller model for faster testing
    config.model.d_model = 8
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.d_ff = 16
    # Use fewer epochs/agents for faster testing
    config.evolution.pretrain_epochs = 1
    config.evolution.evolution_epochs = 1
    config.evolution.num_agents = 2
    config.evolution.num_survivors = 1
    config.model.ltm.d_hidden = 16
    config.model.ltm.num_layers = 1
    config.model.vocab_size = 50

    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size

    train_data = list(range(vocab_size)) * 5
    val_data = list(range(vocab_size)) * 5

    return config, tokenizer, train_data, val_data
