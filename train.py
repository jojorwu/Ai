"""
Основной скрипт для обучения модели Трансформер.
"""
import logging
import os
import time
import numpy as np

from config import Config, TrainingConfig
from data_loader import load_text_from_directory
from model import Transformer
from nn_components.loss import MarginRankingLoss, SoftmaxCrossEntropy
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from utils import load_checkpoint, save_checkpoint
from agent_manager import AgentManager

def setup_logging():
    """Настраивает логирование в файл и в консоль."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_and_prepare_data(config: TrainingConfig):
    """Инициализирует токенизатор и загружает данные для обучения в зависимости от этапа."""
    logging.info("Инициализация токенизатора и загрузка данных...")
    # Токенизатор всегда инициализируется на основных данных для консистентности словаря
    tokenizer = Tokenizer(config.data_dir)

    if config.training_stage == 2:
        data_path = config.sft_data_dir
        logging.info(f"Загрузка данных для этапа 2 (SFT) из: {data_path}")
    else:
        data_path = config.data_dir
        logging.info(f"Загрузка данных для этапа {config.training_stage} из: {data_path}")

    all_text = load_text_from_directory(data_path)
    if not all_text:
        raise ValueError(f"Не удалось загрузить текст из директории: {data_path}")
    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]
    logging.info(f"Данные загружены. Словарь: {tokenizer.vocab_size}. "
                 f"Обучение: {len(train_data)} токенов. Валидация: {len(val_data)}.")
    return tokenizer, train_data, val_data

def initialize_components(config: Config, vocab_size: int):
    """Инициализирует модель, функции потерь и оптимизатор."""
    logging.info("Инициализация модели, функций потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model, ltm_config=config.ltm)
    policy_loss_fn = SoftmaxCrossEntropy()
    value_loss_fn = MarginRankingLoss(margin=config.training.contrastive_margin)

    # Конфигурация основного оптимизатора
    optim_config = config.optimizer.dict()
    max_norm = optim_config.pop('max_norm') # max_norm не является параметром Adam
    optimizer = Adam(**optim_config)

    return model, policy_loss_fn, value_loss_fn, optimizer, max_norm

def get_batches(data, batch_size, seq_len):
    """Генератор батчей для обучения."""
    flat_data = np.array(data, dtype=np.int64)
    num_batches = len(flat_data) // (batch_size * seq_len)
    if num_batches == 0:
        raise ValueError("Недостаточно данных для создания хотя бы одного батча.")
    flat_data = flat_data[:num_batches * batch_size * seq_len]
    x = flat_data.reshape(batch_size, -1)
    y = np.roll(flat_data, -1).reshape(batch_size, -1)
    for i in range(0, x.shape[1], seq_len):
        yield x[:, i:i + seq_len], y[:, i:i + seq_len]

def run_validation(model: Transformer, val_data: list, loss_fn: SoftmaxCrossEntropy, config: TrainingConfig):
    """Запускает валидацию модели на отдельном наборе данных."""
    model.eval()
    total_loss, num_batches = 0, 0
    batch_iterator = get_batches(val_data, config.batch_size, config.seq_len)
    for x, y in batch_iterator:
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)
        logits, _, _ = model.forward(x, mask)
        total_loss += loss_fn.forward(logits, y)
        num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def train_epoch_stage1(model: Transformer, data: list, policy_loss_fn, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху обучения для этапа 1 (только policy pre-training)."""
    train_config, scheduler_config = configs
    start_time = time.time()
    total_policy_loss = 0

    batch_iterator = get_batches(data, train_config.batch_size, train_config.seq_len)
    num_batches = len(data) // (train_config.batch_size * train_config.seq_len)
    training_steps = (num_batches // train_config.gradient_accumulation_steps) * train_config.epochs

    model.zero_grad()
    for i, (x, y) in enumerate(batch_iterator):
        model.train()
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        # Прямой проход
        logits, _, aux_loss = model.forward(x, mask)
        policy_loss = policy_loss_fn.forward(logits, y)

        # Добавляем вспомогательную потерю MoE
        total_loss = policy_loss + train_config.moe_aux_loss_coeff * aux_loss
        total_policy_loss += total_loss

        # Обратный проход
        dlogits = policy_loss_fn.backward()
        # Для dvalues передаем нули, так как value head не используется на этом этапе
        dvalues = np.zeros((x.shape[0], 1))
        model.backward(dlogits, dvalues)

        # Обновление весов с накоплением градиентов
        if (i + 1) % train_config.gradient_accumulation_steps == 0:
            clip_gradients(model.get_named_params(flat=False), max_norm)

            # Передаем max_lr из конфига оптимизатора
            max_lr = optimizer.initial_lr
            new_lr = cosine_decay_with_warmup(current_step, training_steps, max_lr=max_lr, **scheduler_config.dict())
            optimizer.lr = new_lr

            # Собираем параметры и градиенты для шага оптимизатора
            params_with_grads = {}
            named_layers = model.get_named_params()
            for layer_name, layer_obj in named_layers.items():
                if hasattr(layer_obj, 'get_trainable_params'):
                    params_with_grads.update(
                        {f"{layer_name}.{k}": v for k, v in layer_obj.get_trainable_params().items()}
                    )
            optimizer.step(params_with_grads)

            model.zero_grad()
            current_step += 1

    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0
    epoch_time = time.time() - start_time
    return avg_policy_loss, avg_policy_loss, 0.0, epoch_time, current_step


def train_epoch_stage3(model: Transformer, data: list, loss_fns, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху обучения для этапа 3 (contrastive fine-tuning)."""
    train_config, scheduler_config = configs
    policy_loss_fn, value_loss_fn = loss_fns
    start_time = time.time()
    total_policy_loss, total_value_loss = 0, 0

    batch_iterator = get_batches(data, train_config.batch_size, train_config.seq_len)
    num_batches = len(data) // (train_config.batch_size * train_config.seq_len)
    training_steps = (num_batches // train_config.gradient_accumulation_steps) * train_config.epochs

    model.zero_grad()
    for i, (x, y) in enumerate(batch_iterator):
        model.train()

        # Расширение батча для векторизации
        num_candidates = train_config.num_candidates
        original_batch_size = x.shape[0]
        x_expanded = np.repeat(x, num_candidates, axis=0)
        y_expanded = np.repeat(y, num_candidates, axis=0)
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        # Прямой проход для всех кандидатов
        logits, values, aux_loss = model.forward(x_expanded, mask)

        # Векторизованный выбор лучших/худших
        candidate_losses = policy_loss_fn.forward(logits, y_expanded, reduction='none')
        candidate_losses_reshaped = candidate_losses.reshape(original_batch_size, num_candidates)
        best_indices = np.argmin(candidate_losses_reshaped, axis=1)
        worst_indices = np.argmax(candidate_losses_reshaped, axis=1)
        base_indices = np.arange(original_batch_size) * num_candidates
        best_global_indices = base_indices + best_indices
        worst_global_indices = base_indices + worst_indices

        # Расчет потерь
        policy_loss = np.mean(candidate_losses[best_global_indices])
        value_loss = value_loss_fn.forward(values[best_global_indices], values[worst_global_indices])

        # Добавляем вспомогательную потерю MoE
        total_loss = policy_loss + value_loss + train_config.moe_aux_loss_coeff * aux_loss

        total_policy_loss += policy_loss # Отдельно для логирования
        total_value_loss += value_loss   # Отдельно для логирования

        # Расчет и маскирование градиентов
        dlogits = policy_loss_fn.backward()
        policy_mask = np.zeros(x_expanded.shape[0], dtype=bool)
        policy_mask[best_global_indices] = True
        dlogits[~policy_mask] = 0

        d_good_v, d_bad_v = value_loss_fn.backward()
        dvalues = np.zeros_like(values)
        dvalues[best_global_indices] = d_good_v
        dvalues[worst_global_indices] += d_bad_v

        # Обратный проход
        model.backward(dlogits, dvalues)

        # Обновление весов
        if (i + 1) % train_config.gradient_accumulation_steps == 0:
            clip_gradients(model.get_named_params(flat=False), max_norm)

            max_lr = optimizer.initial_lr
            new_lr = cosine_decay_with_warmup(current_step, training_steps, max_lr=max_lr, **scheduler_config.dict())
            optimizer.lr = new_lr

            # Собираем параметры и градиенты для шага оптимизатора
            params_with_grads = {}
            named_layers = model.get_named_params()
            for layer_name, layer_obj in named_layers.items():
                if hasattr(layer_obj, 'get_trainable_params'):
                    params_with_grads.update(
                        {f"{layer_name}.{k}": v for k, v in layer_obj.get_trainable_params().items()}
                    )
            optimizer.step(params_with_grads)

            model.zero_grad()
            current_step += 1

    avg_policy = total_policy_loss / num_batches if num_batches > 0 else 0
    avg_value = total_value_loss / num_batches if num_batches > 0 else 0
    avg_loss = avg_policy + avg_value
    epoch_time = time.time() - start_time
    return avg_loss, avg_policy, avg_value, epoch_time, current_step

def train_epoch_stage4(model: Transformer, tokenizer: Tokenizer, configs):
    """
    Выполняет одну эпоху "агентного" обучения (этап 4).
    """
    train_config, _ = configs
    start_time = time.time()

    # 1. Инициализация менеджера агентов
    # TODO: Сделать количество агентов и др. параметры настраиваемыми
    agent_manager = AgentManager(base_model=model, num_agents=8)

    # 2. Специализация агентов на данных
    agent_manager.specialize_agents(
        data_dir=train_config.data_dir,
        tokenizer=tokenizer,
        tasks_per_agent=10
    )

    # 3. Оценка и отбор лучших
    best_agents = agent_manager.evaluate_and_select_best(top_k=4)

    # 4. Слияние весов LTM
    agent_manager.merge_agents(best_agents)

    epoch_time = time.time() - start_time

    # Для этого этапа возвращаем заглушки, так как "потери" здесь другие
    return 0.0, 0.0, 0.0, epoch_time, 0 # current_step не меняется

from backend import set_backend

def main():
    """Основной скрипт для обучения модели."""
    setup_logging()
    logging.info("--- Запуск обучения модели Трансформер ---")

    # Загрузка конфигурации и установка бэкенда
    config = Config.from_json('config.json')
    set_backend(config.hardware.device)

    tokenizer, train_data, val_data = load_and_prepare_data(config.training)
    model, policy_loss_fn, value_loss_fn, optimizer, max_norm = initialize_components(config, tokenizer.vocab_size)

    start_epoch, current_step = 0, 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.training.checkpoint_path and os.path.exists(config.training.checkpoint_path):
        state, loaded_config = load_checkpoint(model, optimizer, config.training.checkpoint_path)
        if state:
            start_epoch = state.get('epoch', 0)
            current_step = state.get('current_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            epochs_no_improve = state.get('epochs_no_improve', 0)
            logging.info(f"Возобновление с эпохи {start_epoch}, шаг {current_step}.")

    logging.info(f"Начало цикла обучения. Этап: {config.training.training_stage}")
    for epoch in range(start_epoch, config.training.epochs):
        if config.training.training_stage in [1, 2]:
            avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch_stage1(
                model, train_data, policy_loss_fn, optimizer,
                (config.training, config.scheduler), max_norm, current_step
            )
        elif config.training.training_stage == 3:
            avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch_stage3(
                model, train_data, (policy_loss_fn, value_loss_fn), optimizer,
                (config.training, config.scheduler), max_norm, current_step
            )
        elif config.training.training_stage == 4:
            avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch_stage4(
                model, tokenizer, (config.training, config.scheduler)
            )
        else:
            raise ValueError(f"Неизвестный этап обучения: {config.training.training_stage}")

        log_msg_parts = [f"Эпоха {epoch+1}/{config.training.epochs}"]
        if config.training.training_stage == 4:
            log_msg_parts.append("Agent evolution finished")
        else:
            log_msg_parts.extend([
                f"Потери: {avg_loss:.4f}",
                f"(Policy: {avg_policy:.4f}"
            ])
            if config.training.training_stage == 3:
                log_msg_parts.append(f", Value: {avg_value:.4f})")
            else:
                log_msg_parts.append(")")
        log_msg_parts.extend([f"LR: {optimizer.lr:.6f}", f"Время: {epoch_time:.2f}с"])

        val_loss = float('inf')
        if len(val_data) > 0:
            val_loss = run_validation(model, val_data, policy_loss_fn, config.training)
            log_msg_parts.append(f"Val Потери: {val_loss:.4f}")

        logging.info(" | ".join(log_msg_parts))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_weights(config.training.best_model_path, config.dict())
            logging.info(f"Новая лучшая модель сохранена с Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if config.training.checkpoint_path:
            state_to_save = {
                'epoch': epoch + 1,
                'current_step': current_step,
                'best_val_loss': best_val_loss,
                'epochs_no_improve': epochs_no_improve
            }
            save_checkpoint(model, optimizer, state_to_save, config.dict(), config.training.checkpoint_path)

        if epochs_no_improve >= config.training.early_stopping_patience:
            logging.info(f"Ранняя остановка! Нет улучшения Val Loss в течение {epochs_no_improve} эпох.")
            break

    logging.info("Обучение завершено!")
    # Сохраняем финальную модель в любом случае
    model.save_weights(config.training.weights_path, config.dict())

if __name__ == "__main__":
    main()
