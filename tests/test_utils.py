"""
Utilities for setting up tests.
"""
import os
import tempfile
import torch
from peft import PeftModel

from src.config import LoraConfig, TrainConfig
from src.data.tokenizer import Tokenizer
from src.model.loss import cross_entropy_with_label_smoothing
from src.trainer import create_trainer, DataComponents

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


def run_peft_model_test(
    test_case,
    config,
    data_components,
    accelerator,
    load_in_4bit=False,
):
    """
    A generic test function for PEFT models (LoRA, QLoRA).
    - Checks model creation.
    - Verifies parameter efficiency.
    - Runs a single training step.
    - Checks model saving.
    """
    trainer = create_trainer(
        config=config,
        data_components=data_components,
        accelerator=accelerator,
        load_in_4bit=load_in_4bit,
    )

    model = trainer.get_model()
    unwrapped_model = accelerator.unwrap_model(model)
    test_case.assertIsInstance(unwrapped_model, PeftModel)

    total_params = sum(p.numel() for p in unwrapped_model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    test_case.assertLess(trainable_params, total_params)
    test_case.assertGreater(trainable_params, 0)

    input_ids = torch.tensor([[0, 1, 2]], device=accelerator.device)
    labels = torch.tensor([[1, 2, 0]], device=accelerator.device)

    optimizer = trainer.optimizer

    logits, _, _ = model(input_ids=input_ids)

    loss = cross_entropy_with_label_smoothing(
        logits,
        labels,
        smoothing=0.0,
        vocab_size=data_components.tokenizer.vocab_size,
    )

    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

    with tempfile.TemporaryDirectory() as tmpdir:
        unwrapped_model.save_pretrained(tmpdir)
        test_case.assertTrue(
            os.path.exists(os.path.join(tmpdir, "adapter_model.safetensors"))
        )
        test_case.assertTrue(
            os.path.exists(os.path.join(tmpdir, "adapter_config.json"))
        )
