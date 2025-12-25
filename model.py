"""
Main Transformer model implementation.
"""
import json
from dataclasses import dataclass

import numpy as np

from config import DecoderBlockConfig
from nn_components.activations import Tanh
from nn_components.decoder_block import DecoderBlock, ForwardPassInput
from nn_components.embedding import Embedding
from nn_components.kv_cache import KVCache
from nn_components.linear import Linear
from nn_components.long_term_memory import LongTermMemory
from nn_components.rms_norm import RMSNorm
from nn_components.rotary_embedding import precompute_rope_embeddings
from nn_components.utils import softmax
from nn_components.vision_encoder import VisionEncoder
from optimizer import Adam
from utils import zero_gradients


def _sample_from_logits(logits, temperature, top_k, top_p):
    """Performs sampling from logits."""
    if temperature > 0:
        probs = softmax(logits / temperature)
        if top_p > 0.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[1:] = indices_to_remove[:-1].copy()
            indices_to_remove[0] = False
            probs[sorted_indices[indices_to_remove]] = 0
            probs /= np.sum(probs)
        if top_k > 0:
            kth_prob = np.sort(probs)[-top_k]
            probs[probs < kth_prob] = 0
            probs /= np.sum(probs)

        token_id = np.random.choice(len(logits), p=probs)
    else:
        token_id = np.argmax(logits)
    return token_id


# pylint: disable=too-many-instance-attributes
class Transformer:
    """
    Full GPT-style (decoder-only) Transformer model.
    """

    def __init__(self, vocab_size, model_config, vision_config, ltm_config=None, tokenizer=None):
        self.vocab_size = vocab_size
        self.d_model = model_config.d_model
        self.num_layers = model_config.num_layers
        self.num_heads = model_config.num_heads
        self.num_kv_heads = model_config.num_kv_heads
        self.d_ff = model_config.d_ff
        self.max_seq_len = model_config.max_seq_len
        self.dropout_rate = model_config.dropout_rate
        self.ltm_config = ltm_config
        self._flat_params_cache = None
        self.tokenizer = tokenizer

        d_k = self.d_model // self.num_heads
        self.rotary_emb = precompute_rope_embeddings(d_k, self.max_seq_len)
        self.embedding = Embedding(vocab_size, self.d_model)
        self.vision_encoder = VisionEncoder(d_model=self.d_model,
                                            patch_size=vision_config.patch_size,
                                            num_channels=vision_config.num_channels)

        self.long_term_memory = None
        if model_config.ltm_d_hidden and model_config.ltm_num_layers and ltm_config:
            self.long_term_memory = LongTermMemory(self.d_model, model_config.ltm_d_hidden,
                                                   model_config.ltm_num_layers)
            self.ltm_optimizer = Adam(ltm_config.optimizer)
            self.ltm_surprise_threshold = ltm_config.surprise_threshold
        else:
            self.long_term_memory = None

        block_config = DecoderBlockConfig(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            num_kv_heads=self.num_kv_heads,
            rotary_emb=self.rotary_emb,
            num_layers=self.num_layers,
            long_term_memory=self.long_term_memory,
            num_experts=model_config.num_experts,
            top_k_experts=model_config.top_k_experts
        )
        self.decoder_blocks = [DecoderBlock(block_config) for _ in range(self.num_layers)]
        self.final_norm = RMSNorm(self.d_model)
        self.value_head_linear = Linear(self.d_model, 1, bias=False)
        self.value_head_activation = Tanh()
        self.initial_ltm_state = None
        self.final_norm_output = None

    def get_children(self):
        """Returns a dictionary of child layers."""
        children = {'embedding': self.embedding, 'final_norm': self.final_norm,
                    'value_head_linear': self.value_head_linear,
                    'vision_encoder': self.vision_encoder}
        if self.long_term_memory:
            children['long_term_memory'] = self.long_term_memory
        for i, block in enumerate(self.decoder_blocks):
            children[f'decoder_blocks.{i}'] = block
        return children

    def get_named_params(self, obj=None, prefix='', flat=False):
        """
        Recursively collects all trainable layers and their parameters with names.
        Uses caching for a flat list of parameters.
        """
        if flat and self._flat_params_cache is not None:
            return self._flat_params_cache

        if obj is None:
            obj = self

        named_params = {}
        if hasattr(obj, 'get_trainable_params'):
            if flat:
                for param_name, params in obj.get_trainable_params().items():
                    named_params[f"{prefix}.{param_name}"] = params
            else:
                named_params[prefix] = obj

        if hasattr(obj, 'get_children'):
            for name, child in obj.get_children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                named_params.update(self.get_named_params(child, child_prefix, flat=flat))

        if flat and obj is self:
            self._flat_params_cache = named_params

        return named_params

    def count_parameters(self):
        """Counts the total number of trainable parameters in the model."""
        total_params = 0
        # We use flat=True to get a dictionary of individual parameter tensors
        named_params = self.get_named_params(flat=True)
        for _, (param_val, _) in named_params.items():
            total_params += param_val.size
        return total_params

    def zero_grad(self):
        """Resets gradients in all trainable layers to zero."""
        zero_gradients(self)

    def train(self):
        """Switches all layers to training mode."""
        for block in self.decoder_blocks:
            block.train()

    def eval(self):
        """Switches all layers to evaluation (inference) mode."""
        for block in self.decoder_blocks:
            block.eval()

    def get_state(self):
        """Collects the state (weights) of all trainable layers."""
        model_state = {}
        for layer_name, layer_obj in self.get_named_params().items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (param_val, _) in layer_obj.get_trainable_params().items():
                    model_state[f"{layer_name}.{param_name}"] = param_val
        return model_state

    def set_state(self, state_dict):
        """Loads the state (weights) for all trainable layers."""
        for layer_name, layer_obj in self.get_named_params().items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, _ in layer_obj.get_trainable_params().items():
                    load_key = f"{layer_name}.{param_name}"
                    if load_key in state_dict:
                        setattr(layer_obj, param_name, state_dict[load_key])

    def save_weights(self, filepath, config):
        """Saves model weights and configuration to an .npz file."""
        params_to_save = self.get_state()
        config_str = json.dumps(config)
        params_to_save['config'] = np.array([config_str], dtype=object)
        np.savez(filepath, **params_to_save)
        print(f"Model weights and config saved to {filepath}")

    @staticmethod
    def load_model(filepath, vocab_size, config, tokenizer):
        """Loads model weights from an .npz file."""
        model = Transformer(vocab_size=vocab_size,
                            model_config=config.model,
                            vision_config=config.vision,
                            ltm_config=config.ltm,
                            tokenizer=tokenizer)
        with np.load(filepath, allow_pickle=True) as data:
            state_dict = {k: data[k] for k in data if k != 'config'}
            model.set_state(state_dict)
        print(f"Model weights loaded from {filepath}")
        return model

    def _get_embeddings(self, inputs: ForwardPassInput):
        """Gets text and image embeddings."""
        text_embeddings = self.embedding.forward(inputs.x) * np.sqrt(self.d_model)
        if inputs.images is None or self.tokenizer is None:
            return text_embeddings
        image_token_id = self.tokenizer.char_to_idx.get('<IMAGE>')
        if image_token_id is None or np.where(inputs.x == image_token_id)[0].size == 0:
            return text_embeddings
        patch_embeddings = self.vision_encoder.forward(inputs.images)
        final_embeddings = []
        for i in range(inputs.x.shape[0]):
            img_tok_idx = np.where(inputs.x[i] == image_token_id)[0]
            if img_tok_idx.size > 0:
                start_idx = img_tok_idx[0]
                pre_image_part = text_embeddings[i, :start_idx]
                post_image_part = text_embeddings[i, start_idx + 1:]
                combined = np.concatenate([pre_image_part, patch_embeddings[i], post_image_part], axis=0)
                final_embeddings.append(combined)
            else:
                final_embeddings.append(text_embeddings[i])
        return np.array(final_embeddings)

    def _run_decoder_stack(self, h, inputs: ForwardPassInput):
        """Runs the forward pass through the decoder stack."""
        if self.long_term_memory:
            ltm_input = np.mean(h, axis=1, keepdims=True)
            self.initial_ltm_state = self.long_term_memory.forward(ltm_input)
        else:
            self.initial_ltm_state = 0
        current_ltm_state = self.initial_ltm_state
        total_aux_loss = 0
        for i, block in enumerate(self.decoder_blocks):
            forward_pass_input = ForwardPassInput(
                x=h, ltm_state=current_ltm_state, mask=inputs.mask,
                kv_cache=inputs.kv_cache, layer_idx=i, seq_offset=inputs.seq_offset
            )
            h, aux_loss = block.forward(forward_pass_input)
            total_aux_loss += aux_loss
        return h, total_aux_loss

    # pylint: disable=too-many-arguments
    def forward(self, inputs: ForwardPassInput):
        """Performs the forward pass of the model."""
        h = self._get_embeddings(inputs)
        h, total_aux_loss = self._run_decoder_stack(h, inputs)

        h = self.final_norm.forward(h)
        self.final_norm_output = h

        logits = self.final_norm_output @ self.embedding.weights.T
        last_token_hidden_state = h[:, -1, :]
        value_hidden = self.value_head_linear.forward(last_token_hidden_state)
        value = self.value_head_activation.forward(value_hidden)

        return logits, value, total_aux_loss

    def backward(self, dlogits, dvalue):
        """Performs the backward pass of the model."""
        x_norm_reshaped = self.final_norm_output.reshape(-1, self.d_model)
        dlogits_reshaped = dlogits.reshape(-1, self.vocab_size)
        d_embedding_w_from_output = dlogits_reshaped.T @ x_norm_reshaped

        dvalue_hidden = self.value_head_activation.backward(dvalue)
        d_last_token_hidden_state = self.value_head_linear.backward(dvalue_hidden)

        d_h_value = np.zeros_like(self.final_norm_output)
        d_h_value[:, -1, :] = d_last_token_hidden_state
        d_h_policy = dlogits @ self.embedding.weights
        dx = d_h_policy + d_h_value

        dx = self.final_norm.backward(dx)

        total_d_ltm_state = 0
        for block in reversed(self.decoder_blocks):
            dx, d_ltm_state_block = block.backward(dx)
            total_d_ltm_state += d_ltm_state_block

        if self.long_term_memory:
            d_ltm_input = self.long_term_memory.backward(total_d_ltm_state)
            _, seq_len, _ = dx.shape
            dx += d_ltm_input / seq_len

        self.embedding.backward(dx * np.sqrt(self.d_model))
        self.embedding.dweights += d_embedding_w_from_output
        return dx

    def _update_ltm_if_surprised(self, token_arr, kv_cache, seq_offset):
        """Calculates 'surprise' and updates LTM if the threshold is exceeded."""
        if not self.long_term_memory:
            return

        self.zero_grad()
        self.long_term_memory.zero_grad()

        forward_pass_input = ForwardPassInput(
            x=token_arr,
            ltm_state=0,
            kv_cache=kv_cache,
            seq_offset=seq_offset
        )
        logits_for_grad, token_val, _ = self.forward(forward_pass_input)
        d_val = np.ones_like(token_val)
        d_logits = np.zeros_like(logits_for_grad)
        self.backward(d_logits, d_val)

        ltm_params = self.long_term_memory.get_trainable_params()
        squared_grads = [np.sum(grad ** 2) for _, grad in ltm_params.values() if grad is not None]

        if squared_grads:
            grad_norm = np.sqrt(sum(squared_grads))
            if grad_norm > self.ltm_surprise_threshold:
                self.ltm_optimizer.step(ltm_params)

        self.long_term_memory.zero_grad()

    @dataclass
    class SpeculativeChunkInput:
        """Dataclass for storing inputs to the _generate_speculative_chunk method."""
        temp_logits: np.ndarray
        kv_cache: 'KVCache'
        current_seq_len: int
        max_new_tokens: int
        speculative_steps: int
        temperature: float
        top_k: int
        top_p: float

    def _generate_speculative_chunk(self, inputs: SpeculativeChunkInput):
        """Generates a small 'chunk' of tokens speculatively."""
        speculative_chunk = []
        chunk_len = min(inputs.speculative_steps, inputs.max_new_tokens - len(speculative_chunk))
        final_value = None
        temp_logits = inputs.temp_logits

        for i in range(chunk_len):
            token_id = _sample_from_logits(temp_logits[0, -1, :], inputs.temperature, inputs.top_k, inputs.top_p)
            speculative_chunk.append(token_id)
            next_token_arr = np.array([[token_id]])

            self._update_ltm_if_surprised(next_token_arr, inputs.kv_cache, inputs.current_seq_len + i)

            forward_pass_input = ForwardPassInput(
                x=next_token_arr,
                ltm_state=0,
                kv_cache=inputs.kv_cache,
                seq_offset=inputs.current_seq_len + i
            )
            temp_logits, final_value, _ = self.forward(forward_pass_input)
        return speculative_chunk, temp_logits, final_value

    @dataclass
    class GenerateInput:
        """Dataclass for storing inputs to the generate method."""
        start_tokens: np.ndarray
        max_new_tokens: int
        images: np.ndarray = None
        temperature: float = 1.0
        top_k: int = 0
        top_p: float = 0.0
        speculative_steps: int = 5
        value_threshold: float = -1.0
        max_retries: int = 3

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def generate(self, inputs: GenerateInput):
        """Generates a sequence of tokens."""
        self.eval()

        batch_size = 1
        d_k = self.d_model // self.num_heads
        kv_cache = KVCache(self.num_layers, batch_size, self.num_kv_heads, d_k, self.max_seq_len)

        all_generated_tokens = []
        prompt_tokens = np.array(inputs.start_tokens).reshape(batch_size, -1)

        # Initial forward pass for the prompt (can contain an image).
        forward_pass_input = ForwardPassInput(
            x=prompt_tokens,
            ltm_state=0,
            kv_cache=kv_cache,
            seq_offset=0
        )
        logits, _, _ = self.forward(forward_pass_input)

        # Calculate new sequence length if an image was present.
        if (inputs.images is not None and self.tokenizer is not None and
                self.tokenizer.char_to_idx.get('<IMAGE>') in prompt_tokens):
            num_patches = (self.vision_encoder.patch_size // 16) ** 2
            current_seq_len = prompt_tokens.shape[1] - 1 + num_patches
        else:
            current_seq_len = prompt_tokens.shape[1]

        while len(all_generated_tokens) < inputs.max_new_tokens:
            accepted = False
            for _ in range(inputs.max_retries):
                attempt_cache = kv_cache.copy()

                speculative_chunk_input = self.SpeculativeChunkInput(
                    temp_logits=logits,
                    kv_cache=attempt_cache,
                    current_seq_len=current_seq_len,
                    max_new_tokens=inputs.max_new_tokens - len(all_generated_tokens),
                    speculative_steps=inputs.speculative_steps,
                    temperature=inputs.temperature,
                    top_k=inputs.top_k,
                    top_p=inputs.top_p
                )
                chunk, temp_logits, final_value = self._generate_speculative_chunk(speculative_chunk_input)

                if final_value is not None and final_value.item() >= inputs.value_threshold:
                    all_generated_tokens.extend(chunk)
                    current_seq_len += len(chunk)
                    logits = temp_logits
                    kv_cache.restore(attempt_cache.snapshot())
                    accepted = True
                    break

            if not accepted:
                break
        return np.array(all_generated_tokens)
