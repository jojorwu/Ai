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


def create_test_config(num_layers=1, num_experts=None, top_k_experts=None):
    """Creates a minimal TrainConfig with sensible defaults for testing."""
    config_dict = {
        "model": {
            "d_model": 16,
            "num_layers": num_layers,
            "num_heads": 2,
            "d_ff": 32,
            "max_seq_len": 128,
            "dropout_rate": 0.0,
            "num_kv_heads": 2,
            "ltm": {"d_hidden": 16, "num_layers": 1},
            "vocab_size": 50,
            "num_experts": num_experts,
            "top_k_experts": top_k_experts,
            "gradient_checkpointing": False,
        },
        "vision": {"image_size": [224, 224], "patch_size": 16, "num_channels": 3},
        "ltm": {
            "surprise_threshold": 1.0,
            "optimizer": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "hardware": {
            "device": "cpu",
            "strategy": "unified",
            "torch_compile": False,
            "num_threads": 0,
            "num_interop_threads": 0,
            "enable_mkldnn": True,
            "flush_denormals": False,
            "num_workers": 0,
            "pin_memory": False,
        },
        "complexity": {
            "low_complexity_threshold": 0.5,
            "medium_complexity_threshold": 1.5,
            "high_complexity_threshold": 3.0,
            "low_complexity_top_k": 1,
            "medium_complexity_top_k": 1,
            "high_complexity_top_k": 1,
        },
        "evolution": {
            "pretrain_epochs": 1,
            "evolution_epochs": 1,
            "num_agents": 2,
            "num_survivors": 1,
            "batch_size": 2,
            "seq_len": 8,
            "gradient_accumulation_steps": 1,
            "validation_split": 0.1,
            "data_dir": "data",
            "weights_path": "models/test-model/model.pt",
            "early_stopping_patience": 3,
            "best_model_path": "best_model.pt",
            "moe_aux_loss_coeff": 0.01,
            "label_smoothing": 0.0,
        },
        "optimizer": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_norm": 1.0,
        },
        "scheduler": {"warmup_steps": 1, "training_steps": 10, "min_lr": 1e-6},
    }
    return TrainConfig.model_validate(config_dict)


def create_full_test_config_and_data():
    """Creates a minimal configuration and dummy data for testing."""
    config = create_test_config()
    tokenizer = MockTokenizer()
    vocab_size = tokenizer.vocab_size

    train_data = list(range(vocab_size)) * 5
    val_data = list(range(vocab_size)) * 5

    return config, tokenizer, train_data, val_data
