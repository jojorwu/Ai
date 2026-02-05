"""
Handles the initialization of the Transformer model's layers and components.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

import torch
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
from src.model.layers.attention.rotary_embedding import (
    precompute_rope_embeddings,
    precompute_rope_embeddings_2d,
)
from src.model.layers.heads.value_head import ValueHead
from src.model.layers.core.vision import VisionEncoder
from src.model.titans.summary import SummaryNetwork
from src.model.structures import ModelLayers, RopeEmbeddings
from src.utils.exceptions import InitializationError

if TYPE_CHECKING:
    from src.model.model import Transformer


class ModelInitializer:
    """
    Encapsulates the initialization logic for the Transformer model.

    This class handles the creation and registration of embeddings,
    decoder blocks, gating networks, and prediction heads.
    """

    def __init__(self, model: Transformer) -> None:
        """
        Initializes the ModelInitializer.

        Args:
            model: The Transformer model instance to initialize.
        """
        self.model = model
        self.config = model.config
        self.load_in_4bit = model.load_in_4bit

    def init_rope_embeddings(self) -> RopeEmbeddings:
        """
        Initializes and registers Rotary Positional Embeddings (RoPE).
        Supports 2D RoPE for vision tokens prepended to the sequence.

        Returns:
            A RopeEmbeddings dataclass containing cos and sin buffers.
        """
        d_k = self.config.model.d_model // self.config.model.num_heads
        ntk = self.config.model.rope_ntk_factor

        # 1. Precompute 2D RoPE for the vision part
        vh = self.config.vision.image_size[0] // self.config.vision.patch_size
        vw = self.config.vision.image_size[1] // self.config.vision.patch_size

        vis_cos, vis_sin = precompute_rope_embeddings_2d(d_k, vh, vw, ntk_factor=ntk)

        # Add CLS token RoPE (neutral/identity rotation)
        cls_cos = torch.ones(1, 1, d_k // 2)
        cls_sin = torch.zeros(1, 1, d_k // 2)

        vis_cos = torch.cat([cls_cos, vis_cos], dim=0)
        vis_sin = torch.cat([cls_sin, vis_sin], dim=0)

        # 2. Precompute 1D RoPE for the text part
        text_cos, text_sin = precompute_rope_embeddings(
            d_k, self.config.model.max_seq_len, ntk_factor=ntk
        )

        # 3. Concatenate: [Vision RoPE (2D), Text RoPE (1D)]
        # This allows the model to handle multimodal sequences where vision tokens
        # are prepended. Note: pure text sequences will use the beginning of
        # this buffer, which is 2D RoPE. In a production setting, we would
        # handle this indexing more dynamically.
        total_cos = torch.cat([vis_cos, text_cos], dim=0)
        total_sin = torch.cat([vis_sin, text_sin], dim=0)

        self.model.register_buffer("rope_cos_buf", total_cos.clone())
        self.model.register_buffer("rope_sin_buf", total_sin.clone())

        return RopeEmbeddings(cos=self.model.rope_cos_buf, sin=self.model.rope_sin_buf)

    def init_layers(self) -> ModelLayers:
        """
        Initializes all layers of the model.

        Returns:
            A ModelLayers container holding all initialized components.
        """
        ltm = self._init_ltm()
        layers_dict: dict[str, Any] = {}

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

        model_layers = ModelLayers(layers_dict)

        # 4. Apply robust weight initialization
        self._init_weights(model_layers)

        return model_layers

    def _init_weights(self, module: nn.Module) -> None:
        """
        Recursively initializes weights for all sub-modules.

        Uses small standard deviation normal distribution for linear
        and embedding layers, and standard initializers for normalization.

        Args:
            module: The module whose weights need initialization.
        """
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # GPT-2 style small standard deviation for hidden weights
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, VisionEncoder):
                nn.init.trunc_normal_(m.cls_token, std=0.02)
                nn.init.trunc_normal_(m.pos_embed, std=0.02)
            elif isinstance(m, LongTermMemory):
                # Ensure decay bias starts at a reasonable value
                if hasattr(m, "decay_bias"):
                    nn.init.constant_(m.decay_bias, 2.0)
            elif isinstance(m, (nn.LayerNorm, RMSNorm)):
                if hasattr(m, "gamma") and m.gamma is not None:
                    nn.init.ones_(m.gamma)
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _init_core_layers(self, ltm: LongTermMemory | None) -> dict[str, Any]:
        """
        Initializes input, memory, and gating layers.

        Args:
            ltm: Optional LTM module.

        Returns:
            A dictionary of initialized core layers.
        """
        return {
            "embedding": Embedding(self.config.vocab_size, self.config.model.d_model),
            "vision_encoder": VisionEncoder(
                self.config.vision, self.config.model.d_model
            ),
            "long_term_memory": ltm,
            "gating_network": GatingNetwork(
                d_model=self.config.model.d_model,
                num_layers=self.config.model.num_layers,
                num_experts=self.config.model.num_experts,
            ),
            "summary_network": SummaryNetwork(self.config.model.d_model),
            "post_embedding_norm": RMSNorm(self.config.model.d_model),
        }

    def _init_head_layers(self) -> dict[str, Any]:
        """
        Initializes final normalization and prediction heads.

        Returns:
            A dictionary of initialized head layers.
        """
        value_head_linear_class = Linear4bit if self.load_in_4bit else Linear
        return {
            "final_norm": RMSNorm(self.config.model.d_model),
            "value_head": ValueHead(
                self.config.model.d_model, linear_class=value_head_linear_class
            ),
        }

    def _init_ltm(self) -> LongTermMemory | None:
        """
        Initializes the Long-Term Memory (LTM) module if configured.

        Returns:
            An initialized LongTermMemory module or None.

        Raises:
            InitializationError: If the LTM configuration is incomplete.
        """
        cfg = self.config.model.ltm
        if cfg.d_hidden and cfg.num_layers:
            try:
                return LongTermMemory(
                    d_model=self.config.model.d_model,
                    d_hidden=cfg.d_hidden,
                    num_layers=cfg.num_layers,
                    num_heads=cfg.num_heads,
                )
            except Exception as e:
                raise InitializationError(f"Failed to initialize LTM: {e}") from e
        return None

    def _create_decoder_block_config(self, ltm: nn.Module | None) -> DecoderBlockConfig:
        """
        Creates the DecoderBlockConfig for block instantiation.

        Args:
            ltm: The LTM module to share across blocks.

        Returns:
            A DecoderBlockConfig instance.
        """
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
            use_shared_expert=model_cfg.use_shared_expert,
            load_in_4bit=self.load_in_4bit,
        )

    def init_draft_model(self) -> Transformer | None:
        """
        Creates a smaller, faster 'draft' model for speculative decoding.

        Returns:
            An initialized draft Transformer model or None.
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
