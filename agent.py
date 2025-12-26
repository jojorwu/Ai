"""
Implementation of the Agent class for the evolutionary training approach.
"""
import copy
import uuid

from backend import np
from model import GenerateInput, Transformer
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam


class Agent:
    """
    Represents a single "agent" with its own long-term memory (LTM).
    """

    def __init__(self, base_model: Transformer, agent_id=None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.model = copy.deepcopy(base_model)
        self.ltm_optimizer = Adam(self.model.ltm_config.optimizer) if self.model.long_term_memory else None
        self.total_surprise = 0.0
        self.experience_count = 0
        self.value_score_sum = 0.0
        self.loss_fn = SoftmaxCrossEntropy()
        self._fitness_score = -float('inf')

    def experience(self, x_batch: np.ndarray, y_batch: np.ndarray, image_batch: np.ndarray):
        """
        The process of an agent gaining "experience" in a batch training mode.
        """
        if not self.model.long_term_memory or self.ltm_optimizer is None:
            return

        self.model.train()
        self.model.zero_grad()

        logits, values, _ = self.model.forward(x_batch, images=image_batch)
        _ = self.loss_fn.forward(logits, y_batch)
        dlogits = self.loss_fn.backward()
        # Set dvalues to ones to encourage the value head to output higher values,
        # effectively training it to associate seen states with positive outcomes.
        dvalues = np.ones_like(values)
        self.model.backward(dlogits, dvalues)

        ltm_params = self.model.long_term_memory.get_trainable_params()
        if any(p[1] is not None for p in ltm_params.values()):
            self.ltm_optimizer.step(ltm_params)

        self.model.zero_grad()

    def get_ltm_state(self):
        """Returns the state (weights) of this agent's LTM."""
        return self.model.long_term_memory.get_state() if self.model.long_term_memory else None

    def get_fitness_score(self) -> float:
        """Calculates the agent's fitness."""
        return self._fitness_score

    def update_fitness_score(self, score: float):
        """Updates the agent's fitness score."""
        self._fitness_score = score

    def generate_response(self, prompt_tokens: np.ndarray,
                          image_data: np.ndarray = None, max_new_tokens=50) -> np.ndarray:
        """
        Generates a response based on a prompt.
        """
        self.model.eval()
        if prompt_tokens.ndim == 1:
            prompt_tokens = np.expand_dims(prompt_tokens, axis=0)
        images = np.array([image_data]) if image_data is not None else None

        generate_input = GenerateInput(
            start_tokens=prompt_tokens, images=images, max_new_tokens=max_new_tokens,
            temperature=0.7, top_k=50
        )
        # The generate method is now a generator, so we need to consume it.
        generated_tokens = []
        for chunk, _ in self.model.generate(generate_input):
            generated_tokens.extend(chunk.tolist())
        return np.array(generated_tokens)

    def critique_response(self, prompt_tokens: list[int],
                          image_data: np.ndarray, response_tokens: np.ndarray) -> float:
        """
        Evaluates the "usefulness" of a generated response using its Value head.
        """
        self.model.eval()
        full_sequence = np.concatenate([prompt_tokens, response_tokens])
        input_tokens = np.array([full_sequence])
        images = np.array([image_data]) if image_data is not None else None
        _, value, _ = self.model.forward(input_tokens, images=images)
        return value.item() if value is not None else 0.0
