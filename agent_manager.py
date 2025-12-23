"""
Реализация класса AgentManager для управления жизненным циклом агентов.
"""
import copy
from typing import List
from backend import np
from agent import Agent
from model import Transformer
from data_loader import load_text_from_directory
from tokenizer import Tokenizer

class AgentManager:
    """
    Управляет созданием, специализацией и оценкой популяции агентов.
    """
    def __init__(self, base_model: Transformer, num_agents: int):
        """
        Инициализирует менеджер агентов.

        Args:
            base_model (Transformer): "Родительская" модель для клонирования.
            num_agents (int): Количество агентов в популяции.
        """
        self.base_model = base_model
        self.num_agents = num_agents
        self.agents: List[Agent] = []
        self.fork_agents()

    def fork_agents(self):
        """Создает (клонирует) популяцию агентов из базовой модели."""
        print(f"Cloning {self.num_agents} agents from the base model...")
        for i in range(self.num_agents):
            agent = Agent(self.base_model, agent_id=f"agent_{i}")
            self.agents.append(agent)
        print("Agents cloned successfully.")

    def specialize_agents(self, data_dir: str, tokenizer: Tokenizer, tasks_per_agent=5):
        """
        Проводит этап "специализации", где каждый агент получает опыт
        на уникальном подмножестве данных.

        Args:
            data_dir (str): Директория с данными для обучения.
            tokenizer (Tokenizer): Токенизатор для обработки текста.
            tasks_per_agent (int): Сколько фрагментов данных обработает каждый агент.
        """
        print("Specializing agents on different data subsets...")
        # Загружаем все текстовые "документы"
        all_texts = load_text_from_directory(data_dir, as_list=True)
        if not all_texts:
            print("Warning: No data found for agent specialization.")
            return

        # Распределяем задачи (тексты) между агентами
        for i, agent in enumerate(self.agents):
            print(f"  - Specializing {agent.agent_id}...")
            # Выбираем для каждого агента свой набор задач
            agent_task_indices = np.random.choice(len(all_texts), tasks_per_agent, replace=False)

            for task_idx in agent_task_indices:
                text_chunk = all_texts[task_idx]
                tokens = tokenizer.encode(text_chunk)
                if not tokens:
                    continue
                # Агент получает опыт и обновляет свой LTM
                agent.experience(np.array(tokens))
        print("Agent specialization complete.")

    def evaluate_and_select_best(self, top_k: int) -> List[Agent]:
        """
        Оценивает всех агентов и возвращает `top_k` лучших.

        Args:
            top_k (int): Количество лучших агентов для отбора.

        Returns:
            List[Agent]: Список `top_k` лучших агентов.
        """
        print(f"Evaluating agents and selecting top {top_k}...")
        if not self.agents:
            return []

        # Сортируем агентов по их "приспособленности" (fitness score)
        sorted_agents = sorted(self.agents, key=lambda a: a.get_fitness_score(), reverse=True)

        # Возвращаем лучших
        top_agents = sorted_agents[:top_k]
        for agent in top_agents:
            print(f"  - Selected {agent.agent_id} with score: {agent.get_fitness_score():.4f}")
        return top_agents

    def merge_agents(self, best_agents: List[Agent]):
        """
        Усредняет веса LTM "лучших" агентов и обновляет LTM базовой модели.

        Args:
            best_agents (List[Agent]): Список лучших агентов, отобранных
                                     методом `evaluate_and_select_best`.
        """
        if not best_agents:
            print("Warning: No best agents to merge.")
            return

        print(f"Merging LTM weights from {len(best_agents)} best agents...")

        # Получаем состояния LTM от всех лучших агентов
        ltm_states = [agent.get_ltm_state() for agent in best_agents if agent.get_ltm_state() is not None]
        if not ltm_states:
            print("Warning: None of the best agents had a valid LTM state.")
            return

        # Инициализируем усредненное состояние LTM на основе первого агента
        avg_ltm_state = copy.deepcopy(ltm_states[0])

        # Усредняем веса по всем слоям LTM
        for layer_name in avg_ltm_state:
            # Суммируем веса W и b от всех агентов
            sum_W = sum(state[layer_name]['W'] for state in ltm_states)
            sum_b = sum(state[layer_name]['b'] for state in ltm_states if 'b' in state[layer_name])

            # Делим на количество агентов, чтобы получить среднее
            avg_ltm_state[layer_name]['W'] = sum_W / len(ltm_states)
            if 'b' in avg_ltm_state[layer_name]:
                avg_ltm_state[layer_name]['b'] = sum_b / len(ltm_states)

        # Обновляем LTM базовой модели
        if self.base_model.long_term_memory:
            self.base_model.long_term_memory.set_state(avg_ltm_state)
            print("Base model's LTM has been updated with merged weights.")
