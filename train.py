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
    """Инициализирует токенизатор и загружает данные для обучения."""
    logging.info("Инициализация токенизатора и загрузка данных...")
    tokenizer = Tokenizer(config.data_dir)

    # Используем новый data_loader для извлечения текста
    all_text = load_text_from_directory(config.data_dir)

    if not all_text:
        raise ValueError("Не удалось загрузить текст из директории. Убедитесь, что в ней есть файлы .txt, .pdf или .docx.")

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info("Данные успешно загружены. Размер словаря: %d.", tokenizer.vocab_size)
    logging.info("Обучающая выборка: %d токенов. Валидационная: %d.", len(train_data), len(val_data))
    return tokenizer, train_data, val_data

def initialize_components(config: Config, vocab_size: int):
    """Инициализирует модель, функции потерь и оптимизатор."""
    logging.info("Инициализация модели, функций потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, **config.model.dict())
    policy_loss_fn = SoftmaxCrossEntropy()
    value_loss_fn = MarginRankingLoss(margin=config.training.contrastive_margin)

    optim_config = config.optimizer.dict()
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(model.get_named_params(), **optim_config)

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
        logits, _ = model.forward(x, mask)
        total_loss += loss_fn.forward(logits, y)
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0

def _accumulate_gradients(base_grads, new_grads):
    """Рекурсивно добавляет новые градиенты к существующим."""
    for key in base_grads:
        if key in new_grads and new_grads[key] is not None:
            base_grads[key] += new_grads[key]
    return base_grads

def train_epoch(model: Transformer, data: list, loss_fns, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху обучения."""
    train_config, scheduler_config = configs
    policy_loss_fn, value_loss_fn = loss_fns

    start_time = time.time()
    total_loss, total_policy_loss, total_value_loss = 0, 0, 0

    batch_iterator = get_batches(data, train_config.batch_size, train_config.seq_len)
    num_batches = len(data) // (train_config.batch_size * train_config.seq_len)
    training_steps = (num_batches // train_config.gradient_accumulation_steps) * train_config.epochs

    grad_accumulator = {name: np.zeros_like(param) for name, (param, _) in model.get_named_params(flat=True).items()}

    for i, (x, y) in enumerate(batch_iterator):
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        # --- ИСПРАВЛЕНИЕ ОШИБКИ ВЫЧИСЛЕНИЯ ГРАДИЕНТА ---

        # 1. Фаза выбора: генерируем кандидатов в режиме обучения (со стохастичностью, например, dropout)
        #    и находим лучшего и худшего по policy loss.
        model.train()
        candidate_losses = []
        candidate_values = []
        for _ in range(train_config.num_candidates):
            logits, value = model.forward(x, mask)
            loss = policy_loss_fn.forward(logits, y)
            candidate_losses.append(loss)
            candidate_values.append(value)

        best_idx = np.argmin(candidate_losses)
        worst_idx = np.argmax(candidate_losses)
        best_value = candidate_values[best_idx]
        worst_value = candidate_values[worst_idx]

        # 2. Фаза вычисления градиентов:
        #    Выполняем ОДИН новый forward pass в режиме обучения. Градиенты будут считаться
        #    для этого конкретного прохода, обеспечивая консистентность состояния модели.
        model.train()
        logits, _ = model.forward(x, mask)

        # 3. Расчет policy loss и value loss.
        #    Policy loss считается для нового прохода.
        #    Value loss использует значения лучшего/худшего из фазы выбора.
        policy_loss = policy_loss_fn.forward(logits, y)
        dlogits = policy_loss_fn.backward()

        value_loss = value_loss_fn.forward(best_value, worst_value)
        d_good_v, d_bad_v = value_loss_fn.backward()

        # 4. Backward pass для policy loss и "хорошего" value loss.
        #    Состояние модели (dropout маски и т.д.) соответствует `logits`,
        #    для которых был вычислен `dlogits`, что исправляет ошибку.
        model.zero_grad()
        model.backward(dlogits, d_good_v)
        final_grads = model.get_gradients(flat=True)

        # 5. Backward pass для "плохого" value loss.
        #    Нужен еще один forward pass, чтобы установить состояние для этого.
        model.train()
        _ = model.forward(x, mask)
        model.zero_grad()
        model.backward(np.zeros_like(dlogits), d_bad_v)
        final_grads = _accumulate_gradients(final_grads, model.get_gradients(flat=True))

        grad_accumulator = _accumulate_gradients(grad_accumulator, final_grads)

        total_loss += (policy_loss + value_loss)
        total_policy_loss += policy_loss
        total_value_loss += value_loss

        # 6. Обновление весов (остается без изменений)
        if (i + 1) % train_config.gradient_accumulation_steps == 0:
            for key, val in grad_accumulator.items():
                if val is not None:
                    grad_accumulator[key] = val / train_config.gradient_accumulation_steps

            model.set_gradients(grad_accumulator, flat=True)
            clip_gradients(model.get_named_params(flat=False), max_norm)

            new_lr = cosine_decay_with_warmup(current_step, training_steps, **scheduler_config.dict())
            optimizer.lr = new_lr
            optimizer.step()

            model.zero_grad()
            grad_accumulator = {key: np.zeros_like(val) for key, val in grad_accumulator.items()}
            current_step += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_policy = total_policy_loss / num_batches if num_batches > 0 else 0
    avg_value = total_value_loss / num_batches if num_batches > 0 else 0
    epoch_time = time.time() - start_time

    return avg_loss, avg_policy, avg_value, epoch_time, current_step

def main():
    """Основной скрипт для обучения модели."""
    setup_logging()
    logging.info("--- Запуск обучения модели Трансформер ---")

    config = Config.from_json('config.json')

    tokenizer, train_data, val_data = load_and_prepare_data(config.training)
    model, policy_loss_fn, value_loss_fn, optimizer, max_norm = initialize_components(config, tokenizer.vocab_size)

    start_epoch, current_step = 0, 0
    if config.training.checkpoint_path and os.path.exists(config.training.checkpoint_path):
        logging.info("Загрузка с контрольной точки: %s", config.training.checkpoint_path)
        state, _ = load_checkpoint(model, optimizer, config.training.checkpoint_path)
        if state:
            start_epoch, current_step = state['epoch'], state['current_step']
            logging.info("Обучение возобновлено с эпохи %d, шаг %d.", start_epoch, current_step)

    logging.info("Начало цикла обучения...")
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        avg_loss, avg_policy, avg_value, epoch_time, current_step = train_epoch(
            model, train_data, (policy_loss_fn, value_loss_fn), optimizer,
            (config.training, config.scheduler), max_norm, current_step
        )

        log_msg = (f"Эпоха {epoch+1}/{config.training.epochs} | Потери: {avg_loss:.4f} "
                   f"(Policy: {avg_policy:.4f}, Value: {avg_value:.4f}) | "
                   f"LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")

        if len(val_data) > 0:
            val_loss = run_validation(model, val_data, policy_loss_fn, config.training)
            log_msg += f" | Val Потери: {val_loss:.4f}"

        logging.info(log_msg)

        if config.training.checkpoint_path:
            save_checkpoint(model, optimizer, epoch, current_step, config.dict(), config.training.checkpoint_path)

    logging.info("Обучение завершено!")
    model.save_weights(config.training.weights_path, config.dict())

if __name__ == "__main__":
    main()
