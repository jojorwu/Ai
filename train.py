"""
Основной скрипт для агентно-центричного обучения модели Трансформер.
"""
import logging
import os
import time
import numpy as np

from config import Config, EvolutionConfig
from data_loader import load_text_from_directory
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from nn_components.lr_scheduler import cosine_decay_with_warmup
from optimizer import Adam, clip_gradients
from tokenizer import Tokenizer
from utils import load_checkpoint, save_checkpoint
from agent_manager import AgentManager
from backend import set_backend

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

def load_and_prepare_data(config: EvolutionConfig, tokenizer_path: str):
    """Инициализирует токенизатор и загружает данные для обучения."""
    logging.info("Инициализация токенизатора и загрузка данных...")
    tokenizer = Tokenizer(tokenizer_path)

    all_text = load_text_from_directory(config.data_dir)
    if not all_text:
        raise ValueError(f"Не удалось загрузить текст из директории: {config.data_dir}")

    data_tokens = tokenizer.encode(all_text, add_special_tokens=True)
    split_idx = int(len(data_tokens) * (1 - config.validation_split))
    train_data, val_data = data_tokens[:split_idx], data_tokens[split_idx:]

    logging.info(f"Данные загружены. Словарь: {tokenizer.vocab_size}. "
                 f"Обучение: {len(train_data)} токенов. Валидация: {len(val_data)}.")
    return tokenizer, train_data, val_data

def initialize_components(config: Config, vocab_size: int):
    """Инициализирует модель, функцию потерь и оптимизатор."""
    logging.info("Инициализация модели, функции потерь и оптимизатора...")
    model = Transformer(vocab_size=vocab_size, model_config=config.model, ltm_config=config.ltm)
    policy_loss_fn = SoftmaxCrossEntropy()

    optim_config = config.optimizer.dict()
    max_norm = optim_config.pop('max_norm')
    optimizer = Adam(**optim_config)

    return model, policy_loss_fn, optimizer, max_norm

def get_batches(data, batch_size, seq_len):
    """Генератор батчей для обучения."""
    flat_data = np.array(data, dtype=np.int64)
    num_batches = len(flat_data) // (batch_size * seq_len)
    if num_batches == 0:
        # Не бросаем ошибку, а возвращаем пустой генератор, чтобы main мог это обработать
        return
    flat_data = flat_data[:num_batches * batch_size * seq_len]
    x = flat_data.reshape(batch_size, -1)
    y = np.roll(flat_data, -1).reshape(batch_size, -1)
    for i in range(0, x.shape[1], seq_len):
        yield x[:, i:i + seq_len], y[:, i:i + seq_len]

def run_validation(model: Transformer, val_data: list, loss_fn: SoftmaxCrossEntropy, config: EvolutionConfig):
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
    return total_loss / num_batches if num_batches > 0 else float('inf')

def train_pretrain_epoch(model, data, loss_fn, optimizer, configs, max_norm, current_step):
    """Выполняет одну эпоху предобучения (стандартное обучение policy-модели)."""
    evo_config, scheduler_config = configs
    start_time = time.time()
    total_policy_loss = 0

    batch_iterator = get_batches(data, evo_config.batch_size, evo_config.seq_len)
    num_batches = len(data) // (evo_config.batch_size * evo_config.seq_len)
    # Общее количество шагов для LR шедулера
    training_steps = (num_batches // evo_config.gradient_accumulation_steps) * evo_config.pretrain_epochs

    model.zero_grad()
    for i, (x, y) in enumerate(batch_iterator):
        model.train()
        mask = np.triu(np.ones((x.shape[1], x.shape[1])), k=1).astype(bool)

        logits, _, aux_loss = model.forward(x, mask)
        policy_loss = loss_fn.forward(logits, y)
        total_loss = policy_loss + evo_config.moe_aux_loss_coeff * aux_loss
        total_policy_loss += total_loss.item()

        dlogits = loss_fn.backward()
        model.backward(dlogits, np.zeros((x.shape[0], 1))) # dvalues не используются

        if (i + 1) % evo_config.gradient_accumulation_steps == 0:
            clip_gradients(model.get_named_params(flat=False), max_norm)

            max_lr = optimizer.initial_lr
            new_lr = cosine_decay_with_warmup(current_step, training_steps, max_lr, **scheduler_config.dict())
            optimizer.lr = new_lr

            params_with_grads = {f"{name}.{k}": v for name, layer in model.get_named_params().items()
                                 if hasattr(layer, 'get_trainable_params')
                                 for k, v in layer.get_trainable_params().items()}
            optimizer.step(params_with_grads)

            model.zero_grad()
            current_step += 1

    avg_loss = total_policy_loss / num_batches if num_batches > 0 else 0
    epoch_time = time.time() - start_time
    return avg_loss, epoch_time, current_step

def run_evolution_cycle(base_model, tokenizer, data, config: Config):
    """Выполняет один полный цикл эволюции: специализация, оценка, слияние."""
    logging.info("--- Начало нового цикла эволюции ---")
    start_time = time.time()
    evo_config = config.evolution

    # 1. Инициализация менеджера агентов
    agent_manager = AgentManager(base_model=base_model, num_agents=evo_config.num_agents)

    # 2. Специализация агентов (Test-Time Training)
    # Мы передаем весь обучающий датасет, менеджер сам его разделит
    logging.info(f"Специализация {evo_config.num_agents} агентов...")
    agent_manager.specialize_agents_on_dataset(
        full_data=data,
        tokenizer=tokenizer,
        seq_len=evo_config.seq_len,
        batch_size=evo_config.batch_size,
        steps_per_agent=10  # Небольшое количество шагов для специализации
    )

    # 3. Оценка и отбор лучших
    logging.info("Оценка и отбор лучших агентов...")
    best_agents = agent_manager.collaborative_evaluation(
        evaluation_data=data[:50],  # Используем небольшую часть данных для оценки
        tokenizer=tokenizer,
        top_k=evo_config.num_survivors
    )
    if not best_agents:
        logging.warning("Не найдено ни одного подходящего агента для слияния. Пропуск слияния.")
        return time.time() - start_time

    # 4. Слияние весов LTM лучших агентов в базовую модель
    logging.info(f"Слияние LTM от {len(best_agents)} лучших агентов в базовую модель...")
    agent_manager.merge_agents(best_agents)

    epoch_time = time.time() - start_time
    logging.info(f"--- Цикл эволюции завершен за {epoch_time:.2f}с ---")
    return epoch_time

def main():
    """Основной скрипт для агентно-центричного обучения."""
    setup_logging()
    logging.info("--- Запуск агентно-центричного обучения ---")

    config = Config.from_json('config.json')
    set_backend(config.hardware.device)

    # Обратите внимание, что токенизатор обучается на data_dir из evolution конфига
    tokenizer, train_data, val_data = load_and_prepare_data(config.evolution, config.evolution.data_dir)
    model, policy_loss_fn, optimizer, max_norm = initialize_components(config, tokenizer.vocab_size)

    start_epoch, current_step = 0, 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    if config.evolution.checkpoint_path and os.path.exists(config.evolution.checkpoint_path):
        state, _ = load_checkpoint(model, optimizer, config.evolution.checkpoint_path)
        if state:
            start_epoch = state.get('epoch', 0)
            current_step = state.get('current_step', 0)
            best_val_loss = state.get('best_val_loss', float('inf'))
            epochs_no_improve = state.get('epochs_no_improve', 0)
            logging.info(f"Возобновление с эпохи {start_epoch}, шаг {current_step}.")

    # --- Фаза 1: Предварительное обучение ---
    if start_epoch < config.evolution.pretrain_epochs:
        logging.info(f"--- Начало фазы предобучения ({config.evolution.pretrain_epochs} эпох) ---")
        for epoch in range(start_epoch, config.evolution.pretrain_epochs):
            avg_loss, epoch_time, current_step = train_pretrain_epoch(
                model, train_data, policy_loss_fn, optimizer,
                (config.evolution, config.scheduler), max_norm, current_step
            )
            val_loss = run_validation(model, val_data, policy_loss_fn, config.evolution)
            logging.info(f"Эпоха Pre-train {epoch+1}/{config.evolution.pretrain_epochs} | "
                         f"Потери: {avg_loss:.4f} | Val Потери: {val_loss:.4f} | "
                         f"LR: {optimizer.lr:.6f} | Время: {epoch_time:.2f}с")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                model.save_weights(config.evolution.best_model_path, config.dict())
            else:
                epochs_no_improve += 1

            if config.evolution.checkpoint_path:
                state = {'epoch': epoch + 1, 'current_step': current_step, 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
                save_checkpoint(model, optimizer, state, config.dict(), config.evolution.checkpoint_path)

            if epochs_no_improve >= config.evolution.early_stopping_patience:
                logging.warning("Ранняя остановка на фазе предобучения!")
                break

        start_epoch = config.evolution.pretrain_epochs # Гарантируем, что перейдем к эволюции

    # --- Фаза 2: Эволюционный цикл ---
    total_evolution_epochs = config.evolution.pretrain_epochs + config.evolution.evolution_epochs
    logging.info(f"\n--- Начало фазы эволюции ({config.evolution.evolution_epochs} циклов) ---")
    for epoch in range(start_epoch, total_evolution_epochs):
        epoch_time = run_evolution_cycle(model, tokenizer, train_data, config)

        # После каждого цикла эволюции валидируем основную модель
        val_loss = run_validation(model, val_data, policy_loss_fn, config.evolution)
        logging.info(f"Эпоха Evolution {epoch+1}/{total_evolution_epochs} | "
                     f"Val Потери базовой модели: {val_loss:.4f} | Время цикла: {epoch_time:.2f}с")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_weights(config.evolution.best_model_path, config.dict())
            logging.info(f"Новая лучшая базовая модель сохранена с Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if config.evolution.checkpoint_path:
            state = {'epoch': epoch + 1, 'current_step': current_step, 'best_val_loss': best_val_loss, 'epochs_no_improve': epochs_no_improve}
            save_checkpoint(model, optimizer, state, config.dict(), config.evolution.checkpoint_path)

        if epochs_no_improve >= config.evolution.early_stopping_patience:
            logging.info("Ранняя остановка на фазе эволюции!")
            break

    logging.info("Обучение завершено!")
    model.save_weights(config.evolution.weights_path, config.dict())

if __name__ == "__main__":
    main()
