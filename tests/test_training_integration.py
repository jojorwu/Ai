"""
Integration test for the training pipeline.
"""
import logging
import unittest

import numpy as np

from config import Config
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam


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
        batch_size = 2
        seq_len = 4

        config = Config.from_json('config.json')
        model_config = config.model
        model_config.d_model = 8
        model_config.num_layers = 1
        model_config.num_heads = 2
        model_config.num_kv_heads = 2
        model_config.d_ff = 16
        model_config.max_seq_len = 5

        model = Transformer(
            vocab_size=vocab_size, model_config=model_config,
            vision_config=config.vision, ltm_config=config.ltm
        )
        x = np.random.randint(0, vocab_size, (batch_size, seq_len))
        y = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

        policy_loss_fn = SoftmaxCrossEntropy()
        optimizer = Adam(config.optimizer)

        initial_state = {k: np.copy(v) for k, v in model.get_state().items()}

        model.train()
        model.zero_grad()
        logits, value, _ = model.forward(x, mask=mask)
        _ = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()
        model.backward(dlogits, np.zeros_like(value))

        params_with_grads = {
            f"{name}.{k}": (v[0], v[1])
            for name, layer in model.get_named_params().items()
            if hasattr(layer, 'get_trainable_params')
            for k, v in layer.get_trainable_params().items()
        }
        optimizer.step(params_with_grads)

        updated_state = model.get_state()

        weights_updated = any(
            not np.allclose(initial_state[k], updated_state[k])
            for k in initial_state
        )

        self.assertTrue(weights_updated, "Weights were not updated after a training step.")
        logging.info("Training Integration test PASSED.")


if __name__ == "__main__":
    unittest.main()
