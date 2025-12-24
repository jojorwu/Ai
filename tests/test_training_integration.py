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
        d_model = 8
        num_layers = 1
        num_heads = 2
        d_ff = 16
        max_seq_len = 5
        batch_size = 2
        seq_len = 4

        model_config = Config.from_json('config.json').model
        model_config.d_model = d_model
        model_config.num_layers = num_layers
        model_config.num_heads = num_heads
        model_config.num_kv_heads = num_heads
        model_config.d_ff = d_ff
        model_config.max_seq_len = max_seq_len

        model = Transformer(vocab_size=vocab_size, model_config=model_config)
        x = np.random.randint(0, vocab_size, (batch_size, seq_len))
        y = np.random.randint(0, vocab_size, (batch_size, seq_len))
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

        policy_loss_fn = SoftmaxCrossEntropy()
        optimizer = Adam(learning_rate=0.001)

        initial_weights = np.copy(model.decoder_blocks[0].moe_layer.experts[0].w1.W)

        model.train()
        model.zero_grad()
        logits, value, _ = model.forward(x, mask=mask)
        _ = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()
        model.backward(dlogits, np.zeros_like(value))

        params_with_grads = {}
        named_layers = model.get_named_params()
        for layer_name, layer_obj in named_layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                params_with_grads.update(
                    {f"{layer_name}.{k}": v for k, v in layer_obj.get_trainable_params().items()}
                )
        optimizer.step(params_with_grads)

        updated_weights = model.decoder_blocks[0].moe_layer.experts[0].w1.W

        self.assertFalse(np.allclose(initial_weights, updated_weights),
                         "Weights were not updated after a training step.")
        logging.info("Training Integration test PASSED.")


if __name__ == "__main__":
    unittest.main()
