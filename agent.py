"""
Реализация класса Agent для эволюционного подхода к обучению.
"""

import copy
import uuid
from backend import np
from model import Transformer
from optimizer import Adam

class Agent:
    """
    Представляет собой одного "мини-агента" с собственной долгосрочной памятью (LTM).
    Агент инкапсулирует модель Transformer и логику для обучения своей LTM
    на основе нового опыта (Test-Time Training).
    """
    def __init__(self, base_model: Transformer, agent_id=None):
        """
        Инициализирует агента, клонируя базовую модель.

        Args:
            base_model (Transformer): "Родительская" модель, чьи веса будут скопированы.
            agent_id (str, optional): Уникальный идентификатор агента.
        """
        self.agent_id = agent_id or str(uuid.uuid4())

        # Глубокое копирование, чтобы у каждого агента была своя независимая модель
        self.model = copy.deepcopy(base_model)

        # У каждого агента свой собственный, уникальный LTM и его оптимизатор.
        # Это ключевая часть его "индивидуальности".
        if self.model.long_term_memory:
            # Re-initialize LTM weights to start fresh
            self.model.long_term_memory.reinitialize_weights()

            # Создаем новый оптимизатор специально для этого LTM
            self.ltm_optimizer = Adam(**self.model.ltm_config.optimizer.dict())
            self.model.ltm_optimizer = self.ltm_optimizer # Привязываем к модели для `generate`

        # Метрики для оценки "успешности" агента
        self.total_surprise = 0.0
        self.experience_count = 0
        self.value_score_sum = 0.0

    def experience(self, data_tokens: np.ndarray, max_len=20):
        """
        Процесс получения "опыта" агентом.
        Модель генерирует ответ на данные, и если "удивление" велико,
        обновляет веса своего LTM.

        Args:
            data_tokens (np.ndarray): Входные данные в виде токенов.
        """
        if not self.model.long_term_memory:
            return

        # Метод generate уже содержит логику Test-Time Training (обновление LTM)
        # Мы просто вызываем его и собираем метрики
        self.model.eval()

        # Мы не используем сгенерированные токены, только запускаем процесс для обновления LTM
        _, avg_surprise, avg_value = self.model.generate(
            start_tokens=data_tokens,
            max_len=max_len,
            # Включаем TTT, но отключаем сложную логику генерации для скорости
            speculative_steps=1,
            temperature=0.0 # Жадная генерация
        )

        # Обновляем метрики агента
        self.total_surprise += avg_surprise
        self.value_score_sum += avg_value
        self.experience_count += 1

    def get_ltm_state(self):
        """Возвращает состояние (веса) LTM этого агента."""
        if self.model.long_term_memory:
            return self.model.long_term_memory.get_state()
        return None

    def get_fitness_score(self) -> float:
        """
        Рассчитывает "приспособленность" агента.
        Комбинированная метрика: среднее "удивление" + средний "value score".
        """
        if self.experience_count == 0:
            return 0.0

        avg_surprise = self.total_surprise / self.experience_count
        avg_value = self.value_score_sum / self.experience_count

        # Комбинируем метрики. Можно добавить веса, если одна важнее другой.
        return avg_surprise + avg_value
