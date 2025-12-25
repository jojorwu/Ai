"""
Utility functions for checkpointing and data batching.
"""
import json
import logging
import os

import numpy as np


def save_checkpoint(model, optimizer, training_state, config, filepath):
    """Saves the state of the model, optimizer, and training."""
    try:
        model_state = model.get_state()
        optimizer_state = optimizer.get_state()

        np_training_state = {key: np.array(value) for key, value in training_state.items()}

        checkpoint = {
            **model_state,
            'optimizer_m': optimizer_state['m'],
            'optimizer_v': optimizer_state['v'],
            'optimizer_t': np.array(optimizer_state['t']),
            **np_training_state
        }

        config_str = json.dumps(config)
        checkpoint['config'] = np.array([config_str], dtype=object)

        np.savez(filepath, **checkpoint)
        logging.info("Checkpoint successfully saved to %s", filepath)

    except (IOError, OSError, KeyError) as e:
        logging.error("Error saving checkpoint to %s: %s", filepath, e, exc_info=True)


def load_checkpoint(model, optimizer, filepath):
    """Loads the state of the model, optimizer, and training."""
    if not os.path.exists(filepath):
        logging.warning("Checkpoint file not found: %s", filepath)
        return None, None

    try:
        data = np.load(filepath, allow_pickle=True)

        model.set_state(data)

        optimizer_state = {
            'm': data['optimizer_m'].item(),
            'v': data['optimizer_v'].item(),
            't': data['optimizer_t'].item()
        }
        optimizer.set_state(optimizer_state)

        training_state = {
            'epoch': data.get('epoch', 0).item(),
            'current_step': data.get('current_step', 0).item(),
            'best_val_loss': data.get('best_val_loss', float('inf')).item(),
            'epochs_no_improve': data.get('epochs_no_improve', 0).item()
        }

        config = json.loads(data['config'][0])

        logging.info("Checkpoint successfully loaded from %s", filepath)
        return training_state, config

    except (IOError, OSError, KeyError, json.JSONDecodeError) as e:
        logging.error("Error loading checkpoint from %s: %s", filepath, e, exc_info=True)
        return None, None


def zero_gradients(model):
    """Recursively zeros out gradients for all trainable parameters in a model."""
    for layer_obj in model.get_named_params().values():
        if hasattr(layer_obj, 'get_trainable_params'):
            for param_name, _ in layer_obj.get_trainable_params().items():
                grad_attr_name = f"d{param_name}"
                if hasattr(layer_obj, grad_attr_name):
                    grad_val = getattr(layer_obj, grad_attr_name)
                    if grad_val is not None:
                        setattr(layer_obj, grad_attr_name, np.zeros_like(grad_val))
