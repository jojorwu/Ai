"""
Factory functions for creating model-related components.
"""
import logging
import os
from typing import Tuple, Union

import torch
from accelerate import dispatch_model

from src.config.core import (GenerateConfig, TrainConfig,
                             TransformerConfig)
from src.data.tokenizer import Tokenizer
from src.model.device_manager import DeviceManager
from src.model.model import Transformer


def load_model_and_tokenizer(
    model_name: str,
    config: Union[TrainConfig, GenerateConfig],
    load_in_4bit: bool = False,
    quantized: bool = False,
    dispatch: bool = True,
) -> Tuple[Transformer, Tokenizer]:
    """Loads a model and tokenizer from a given model name."""
    logging.info(
        "Loading model '%s' (4-bit: %s, quantized: %s)...",
        model_name,
        load_in_4bit,
        quantized,
    )
    model_dir = os.path.join("models", model_name)
    tokenizer = Tokenizer(model_dir)

    vocab_size = config.model.vocab_size or tokenizer.vocab_size
    transformer_config = TransformerConfig(
        vocab_size=vocab_size,
        model=config.model,
        vision=config.vision,
        ltm=config.ltm,
        tokenizer=tokenizer,
    )
    device_manager = DeviceManager(config.hardware)
    device_manager.optimize_environment()
    if device_manager.should_disable_4bit() and load_in_4bit:
        logging.warning("4-bit quantization is not supported on this hardware, disabling.")
        load_in_4bit = False

    model = Transformer(transformer_config, load_in_4bit=load_in_4bit)

    # Determine weights path
    weights_path = None
    if isinstance(config, TrainConfig):
        weights_path = config.evolution.weights_path
    if not weights_path or not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "best_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_dir, "model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights file found in {model_dir} or specified in config.")

    # Optimized weight loading: use mmap if available for faster loading
    # and lower memory peak. Load directly to CPU first to avoid OOM,
    # dispatch_model will handle device placement.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict # Free memory

    if quantized or config.hardware.dynamic_quantization:
        if config.hardware.device == "cpu":
            logging.info("Applying dynamic quantization for CPU...")
            # Quantize both Linear and Embedding layers for better efficiency on CPU.
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8
            )
        else:
            logging.warning("Dynamic quantization is only supported on CPU device. Skipping.")

    if dispatch:
        device_map = device_manager.get_device_map()
        model = dispatch_model(model, device_map=device_map)

    if config.hardware.torch_compile:
        logging.info("Compiling the model with torch.compile...")
        model = torch.compile(model)

    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
    return model, tokenizer
