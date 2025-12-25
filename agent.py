"""
Implementation of the Agent class for the evolutionary training approach.
"""
import copy
import uuid

from backend import np

from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam


class AgentMetrics:
    """A container for agent-specific metrics."""
    def __init__(self):
        self.total_surprise = 0.0
        self.experience_count = 0
        self.value_score_sum = 0.0
        self.fitness_score = -float('inf')


class Agent:
    """
    Represents a single "agent" with its own long-term memory (LTM).
    The agent encapsulates a Transformer model and the logic to train its LTM
    based on new experiences (Test-Time Training).
    """

    def __init__(self, base_model: Transformer, agent_id=None):
        """
        Initializes an agent by cloning a base model.
        Args:
            base_model (Transformer): The "parent" model whose weights will be copied.
            agent_id (str, optional): A unique identifier for the agent.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.model = copy.deepcopy(base_model)

        if self.model.long_term_memory:
            # Re-initialize LTM weights so each agent starts fresh
            self.model.long_term_memory.reinitialize_weights()
            # Each agent gets its own optimizer for its LTM
            self.ltm_optimizer = Adam(**self.model.ltm_config.optimizer.model_dump())
        else:
            self.ltm_optimizer = None

        self.loss_fn = SoftmaxCrossEntropy()
        self.metrics = AgentMetrics()

    def experience(self, x_batch: np.ndarray, y_batch: np.ndarray):
        """
        The process of an agent gaining "experience" in a batch training mode.
        Performs one LTM update step if the "surprise" is large enough.
        """
        if not self.model.long_term_memory or self.ltm_optimizer is None:
            return

        self.model.train()
        self.model.zero_grad()

        # 1. Forward and backward pass to get gradients
        logits, values, _ = self.model.forward(x_batch)
        _ = self.loss_fn.forward(logits, y_batch)
        dlogits = self.loss_fn.backward()
        dvalues = np.zeros_like(values)
        self.model.backward(dlogits, dvalues)

        # 2. Extract LTM gradients and calculate "surprise"
        ltm_params = self.model.long_term_memory.get_trainable_params()
        flat_grads = np.concatenate([
            p['grad'].ravel() for p in ltm_params.values() if p['grad'] is not None
        ])

        surprise = np.linalg.norm(flat_grads) if flat_grads.size > 0 else 0.0

        # 3. Update LTM weights if surprise is high
        if surprise > self.model.ltm_config.surprise_threshold:
            self.ltm_optimizer.step(ltm_params)

        # 4. Update agent metrics
        self.metrics.total_surprise += surprise
        self.metrics.value_score_sum += np.mean(values)
        self.metrics.experience_count += 1

        self.model.zero_grad()

    def get_ltm_state(self):
        """Returns the state (weights) of this agent's LTM."""
        if self.model.long_term_memory:
            return self.model.long_term_memory.get_state()
        return None

    def get_fitness_score(self) -> float:
        """
        Calculates the agent's fitness.
        This score is now set externally by the AgentManager after cross-critique.
        """
        return self.metrics.fitness_score

    def update_fitness_score(self, score: float):
        """Updates the agent's fitness score (used by AgentManager)."""
        self.metrics.fitness_score = score

    def generate_response(self, prompt_tokens: np.ndarray,
                          image_data: np.ndarray = None, max_new_tokens=50) -> list[int]:
        """
        Generates a response based on a prompt (text + image).
        """
        self.model.eval()

        if prompt_tokens.ndim == 1:
            prompt_tokens = np.expand_dims(prompt_tokens, axis=0)

        images = np.array([image_data]) if image_data is not None else None

        generated_tokens = self.model.generate(
            start_tokens=prompt_tokens,
            images=images,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50
        )

        return generated_tokens.tolist()

    def critique_response(self, prompt_tokens: list[int],
                          image_data: np.ndarray, response_tokens: list[int]) -> float:
        """
        Evaluates the "usefulness" of a generated response using its Value head.
        """
        self.model.eval()
        full_sequence = prompt_tokens + response_tokens
        input_tokens = np.array([full_sequence])

        _logits, value, _aux_loss = self.model.forward(
            input_tokens,
            images=np.array([image_data]) if image_data is not None else None
        )

        return value.item() if value is not None else 0.0
