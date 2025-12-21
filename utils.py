import numpy as np
import json
import os
import logging

def save_checkpoint(model, optimizer, epoch, current_step, config, filepath):
    """Сохраняет состояние модели, оптимизатора и обучения."""
    try:
        model_state = model.get_state()
        optimizer_state = optimizer.get_state()

        training_state = {
            'epoch': np.array(epoch),
            'current_step': np.array(current_step)
        }

        checkpoint = {
            **model_state,
            'optimizer_m': optimizer_state['m'],
            'optimizer_v': optimizer_state['v'],
            'optimizer_t': np.array(optimizer_state['t']),
            **training_state
        }

        config_str = json.dumps(config)
        checkpoint['config'] = np.array([config_str], dtype=object)

        np.savez(filepath, **checkpoint)
        logging.info(f"Контрольная точка успешно сохранена в {filepath}")

    except Exception as e:
        logging.error(f"Ошибка при сохранении контрольной точки в {filepath}: {e}", exc_info=True)

def load_checkpoint(model, optimizer, filepath):
    """Загружает состояние модели, оптимизатора и обучения."""
    if not os.path.exists(filepath):
        logging.warning(f"Файл контрольной точки не найден: {filepath}")
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
            'epoch': data['epoch'].item(),
            'current_step': data['current_step'].item()
        }

        config = json.loads(data['config'][0])

        logging.info(f"Контрольная точка успешно загружена из {filepath}")
        return training_state, config

    except Exception as e:
        logging.error(f"Ошибка при загрузке контрольной точки из {filepath}: {e}", exc_info=True)
        return None, None
