"""
Handles the initialization of the Transformer model's layers and components.
"""
import logging
from typing import TYPE_CHECKING

from bitsandbytes.nn import Linear4bit
from torch import nn

from src.config.core import TransformerConfig
from src.config.model_config import DecoderBlockConfig
from src.model.layers.blocks.decoder_block import DecoderBlock
from src.model.layers.core.embedding import Embedding
from src.model.layers.titans.gating import GatingNetwork
from src.model.layers.core.linear import Linear
from src.model.layers.titans.long_term_memory import LongTermMemory
from src.model.layers.core.rms_norm import RMSNorm
from src.model.layers.attention.rotary_embedding import precompute_rope_embeddings
from src.model.layers.heads.value_head import ValueHead
from src.model.structures import ModelLayers, RopeEmbeddings

if TYPE_CHECKING:
    from src.model.model import Transformer


class ModelInitializer:
    """
    Encapsulates the initialization logic for the Transformer model.
    """

    def __init__(self, model: "Transformer") -> None:
        self.model = model
        self.config = model.config
        self.load_in_4bit = model.load_in_4bit

    def init_rope_embeddings(self) -> RopeEmbeddings:
        """Initializes and registers Rotary Positional Embeddings (RoPE)."""
        d_k = self.config.model.d_model // self.config.model.num_heads
        rope_cos, rope_sin = precompute_rope_embeddings(
            d_k,
            self.config.model.max_seq_len,
            ntk_factor=self.config.model.rope_ntk_factor
        )
        self.model.register_buffer("rope_cos_buf", rope_cos.clone())
        self.model.register_buffer("rope_sin_buf", rope_sin.clone())
        return RopeEmbeddings(cos=self.model.rope_cos_buf, sin=self.model.rope_sin_buf)

    def init_layers(self) -> ModelLayers:
        """
        Initializes all layers of the model, organizing them into core and head layers.
        """
        ltm = self._init_ltm()
        layers_dict = {}

        # 1. Initialize core input and routing layers
        layers_dict.update(self._init_core_layers(ltm))

        # 2. Initialize the stack of decoder blocks
        decoder_config = self._create_decoder_block_config(ltm)
        layers_dict["decoder"] = nn.ModuleList(
            [
                DecoderBlock(decoder_config)
                for _ in range(self.config.model.num_layers)
            ]
        )

        # 3. Initialize output and value heads
        layers_dict.update(self._init_head_layers())

        return ModelLayers(layers_dict)

    def _init_core_layers(self, ltm: LongTermMemory | None) -> dict:
        """Initializes input, memory, and gating layers."""
        return {
            "embedding": Embedding(self.config.vocab_size, self.config.model.d_model),
            "long_term_memory": ltm,
            "gating_network": GatingNetwork(
                d_model=self.config.model.d_model,
                num_layers=self.config.model.num_layers,
                num_experts=self.config.model.num_experts,
            ),
        }

    def _init_head_layers(self) -> dict:
        """Initializes final normalization and prediction heads."""
        value_head_linear_class = Linear4bit if self.load_in_4bit else Linear
        return {
            "final_norm": RMSNorm(self.config.model.d_model),
            "value_head": ValueHead(
                self.config.model.d_model, linear_class=value_head_linear_class
            ),
        }

    def _init_ltm(self) -> LongTermMemory | None:
        """Initializes the Long-Term Memory (LTM) module if configured."""
        cfg = self.config.model.ltm
        if cfg.d_hidden and cfg.num_layers:
            return LongTermMemory(
                d_model=self.config.model.d_model,
                d_hidden=cfg.d_hidden,
                num_layers=cfg.num_layers,
            )
        return None

    def _create_decoder_block_config(self, ltm: nn.Module) -> DecoderBlockConfig:
        """Helper method to create the DecoderBlockConfig."""
        model_cfg = self.config.model
        return DecoderBlockConfig(
            d_model=model_cfg.d_model,
            num_heads=model_cfg.num_heads,
            d_ff=model_cfg.d_ff,
            dropout_rate=model_cfg.dropout_rate,
            num_kv_heads=model_cfg.num_kv_heads,
            rotary_emb=(self.model.rope_embeddings.cos, self.model.rope_embeddings.sin),
            num_layers=model_cfg.num_layers,
            long_term_memory=ltm,
            num_experts=model_cfg.num_experts,
            top_k_experts=model_cfg.top_k_experts,
            load_in_4bit=self.load_in_4bit,
        )

    def init_draft_model(self) -> "Transformer|None":
        """
        Creates a smaller, faster 'draft' model for speculative decoding.
        """
        if self.config.model.num_layers < 2:
            return None

        draft_config_dict = self.config.model_dump()
        draft_config_dict["model"]["num_layers"] //= 2
        draft_config = TransformerConfig(**draft_config_dict)

        logging.info(
            "Creating a draft model with %d layers.",
            draft_config.model.num_layers,
        )
        return self.model.__class__(draft_config, self.load_in_4bit)
