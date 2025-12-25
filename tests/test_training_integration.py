"""
Integration test for the training pipeline.
"""
import logging
import unittest

import numpy as np

from config import Config
from model import Transformer, ForwardPassInput
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam


def _create_test_config():
    """Creates a minimal configuration for integration testing."""
    config = Config.from_json('config.json')
    config.model.d_model = 8
    config.model.num_layers = 1
    config.model.num_heads = 2
    config.model.num_kv_heads = 2
    config.model.d_ff = 16
    config.model.max_seq_len = 5
    config.evolution.batch_size = 2
    config.evolution.seq_len = 4
    return config


class TestTrainingIntegration(unittest.TestCase):
    """
    Tests that a single training step updates the model's weights.
    This is a smoke test for the entire training pipeline.
    """

    def test_single_training_step(self):
        """
        Tests that a single, simple training step updates the model's weights.
        """
        logging.info("\nRunning Test: Training Integration (single step)...")
        vocab_size = 10
        config = _create_test_config()

        model = Transformer(vocab_size=vocab_size,
                            model_config=config.model,
                            vision_config=config.vision,
                            ltm_config=config.ltm)
        x = np.random.randint(0, vocab_size, (config.evolution.batch_size, config.evolution.seq_len))
        y = np.random.randint(0, vocab_size, (config.evolution.batch_size, config.evolution.seq_len))
        mask = np.triu(np.ones((config.evolution.seq_len, config.evolution.seq_len)), k=1).astype(bool)

        policy_loss_fn = SoftmaxCrossEntropy()
        optimizer = Adam(config.optimizer)

        initial_state = {k: np.copy(v) for k, v in model.get_state().items()}

        model.train()
        model.zero_grad()
        forward_input = ForwardPassInput(x=x, ltm_state=0, mask=mask)
        logits, value, _ = model.forward(forward_input)
        _ = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()
        model.backward(dlogits, np.zeros_like(value))

        params_with_grads = {f"{name}.{k}": (v[0], v[1]) for name, layer in model.get_named_params().items()
                                     if hasattr(layer, 'get_trainable_params')
                                     for k, v in layer.get_trainable_params().items()}
        optimizer.step(params_with_grads)

        updated_state = model.get_state()

        weights_updated = any(not np.allclose(initial_state[k], updated_state[k]) for k in initial_state)

        self.assertTrue(weights_updated, "Weights were not updated after a training step.")
        logging.info("Training Integration test PASSED.")


if __name__ == "__main__":
    unittest.main()
