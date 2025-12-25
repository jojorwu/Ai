"""
Utility functions for checkpointing and data batching.
"""
import json
import logging
import os

import numpy as np


# pylint: disable=broad-except-in-catch
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

    except Exception as e:
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
            'epoch': data['epoch'].item() if 'epoch' in data else 0,
            'current_step': data['current_step'].item() if 'current_step' in data else 0,
            'best_val_loss': data['best_val_loss'].item() if 'best_val_loss' in data else float('inf'),
            'epochs_no_improve': data['epochs_no_improve'].item() if 'epochs_no_improve' in data else 0
        }

        config = json.loads(data['config'][0])

        logging.info("Checkpoint successfully loaded from %s", filepath)
        return training_state, config

    except Exception as e:
        logging.error("Error loading checkpoint from %s: %s", filepath, e, exc_info=True)
        return None, None

def get_batches(data, batch_size, seq_len, shuffle=False):
    """
    Generator function to yield batches of data. Handles both text-only and multimodal data.
    """
    if not data:
        return

    if shuffle:
        np.random.shuffle(data)

    # Check if data is multimodal (list of tuples) or text-only (flat list)
    is_multimodal = isinstance(data[0], (list, tuple))

    num_sequences = len(data) - seq_len
    for i in range(0, num_sequences, batch_size):
        batch_end = i + batch_size
        x_list, y_list, img_list = [], [], []

        for j in range(i, min(batch_end, num_sequences)):
            if is_multimodal:
                # Unpack tuples for multimodal data
                x_seq = [item[0] for item in data[j:j + seq_len]]
                y_seq = [item[0] for item in data[j + 1:j + seq_len + 1]]
                img_seq = [item[1] for item in data[j:j + seq_len]]
                x_list.append(x_seq)
                y_list.append(y_seq)
                img_list.append(img_seq)
            else:
                # Handle text-only data
                x_list.append(data[j:j + seq_len])
                y_list.append(data[j + 1:j + seq_len + 1])

        if is_multimodal:
            yield np.array(x_list), np.array(y_list), np.array(img_list)
        else:
            yield np.array(x_list), np.array(y_list)

def zero_gradients(model):
    """Resets gradients in all trainable layers of a model to zero."""
    for layer_obj in model.get_named_params().values():
        if hasattr(layer_obj, 'get_trainable_params'):
            for param_name, _ in layer_obj.get_trainable_params().items():
                grad_attr_name = f"d{param_name}"
                if hasattr(layer_obj, grad_attr_name):
                    grad_val = getattr(layer_obj, grad_attr_name)
                    if grad_val is not None:
                        setattr(layer_obj, grad_attr_name, np.zeros_like(grad_val))
