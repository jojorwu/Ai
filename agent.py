"""
Implementation of the Agent class for the evolutionary training approach.
"""
import copy
import uuid
from dataclasses import dataclass, field

from backend import np
from model import GenerateInput, Transformer
from nn_components.decoder_block import ForwardPassInput
from nn_components.loss import SoftmaxCrossEntropy
from optimizer import Adam


@dataclass
class AgentMetrics:
    """Keeps track of an agent's performance metrics."""
    total_surprise: float = 0.0
    experience_count: int = 0
    value_score_sum: float = 0.0
    fitness_score: float = field(default=-float('inf'))


class Agent:
    """
    Represents a single "agent" with its own long-term memory (LTM).
    """

    def __init__(self, base_model: Transformer, agent_id=None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.model = copy.deepcopy(base_model)
        if self.model.long_term_memory:
            self.ltm_optimizer = Adam(self.model.config.ltm.optimizer)
        else:
            self.ltm_optimizer = None
        self.loss_fn = SoftmaxCrossEntropy()
        self.metrics = AgentMetrics()

    def _forward_pass(self, x_batch, y_batch, image_batch):
        forward_pass_input = ForwardPassInput(x=x_batch, images=image_batch, ltm_state=0)
        logits, values, _ = self.model.forward(forward_pass_input)
        logits_for_loss = logits[:, 1:, :] if self.model.long_term_memory else logits
        self.loss_fn.forward(logits_for_loss, y_batch)
        return logits, values

    def _backward_pass(self, logits):
        dlogits_from_loss = self.loss_fn.backward()
        if self.model.long_term_memory:
            batch_size, _, vocab_size = logits.shape
            padding = np.zeros((batch_size, 1, vocab_size), dtype=dlogits_from_loss.dtype)
            dlogits = np.concatenate([padding, dlogits_from_loss], axis=1)
        else:
            dlogits = dlogits_from_loss
        return dlogits

    def _update_ltm(self):
        ltm_params = self.model.long_term_memory.get_trainable_params()
        if any(p[1] is not None for p in ltm_params.values()):
            flat_grads = np.concatenate([p[1].ravel() for p in ltm_params.values() if p[1] is not None])
            surprise = np.linalg.norm(flat_grads) if flat_grads.size > 0 else 0.0
            self.metrics.total_surprise += surprise
            if surprise > self.model.ltm_surprise_threshold:
                self.ltm_optimizer.step(ltm_params)

    def experience(self, x_batch: np.ndarray, y_batch: np.ndarray, image_batch: np.ndarray):
        """
        The process of an agent gaining "experience" in a batch training mode.
        """
        if not self.model.long_term_memory or self.ltm_optimizer is None:
            return

        self.model.train()
        self.model.zero_grad()

        logits, values = self._forward_pass(x_batch, y_batch, image_batch)
        dlogits = self._backward_pass(logits)
        dvalues = np.ones_like(values)
        self.model.backward(dlogits, dvalues)

        self._update_ltm()

        self.metrics.value_score_sum += np.mean(values)
        self.metrics.experience_count += 1
        self.model.zero_grad()

    def get_ltm_state(self):
        """Returns the state (weights) of this agent's LTM."""
        return self.model.long_term_memory.get_state() if self.model.long_term_memory else None

    def get_fitness_score(self) -> float:
        """Calculates the agent's fitness."""
        return self.metrics.fitness_score

    def update_fitness_score(self, score: float):
        """Updates the agent's fitness score."""
        self.metrics.fitness_score = score

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
