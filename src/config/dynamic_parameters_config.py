"""
Pydantic models for text generation configuration.
"""
from pydantic import BaseModel, Field


class DynamicParametersConfig(BaseModel):
    """Configuration for the dynamic parameters of the text generation process."""
    start_text: str = Field(...,
                            description="Initial text for generation.")
    max_len: int = Field(...,
                         description="Maximum length of the generated text.")
    temperature: float = Field(..., description="Temperature for sampling.")
    top_k: int = Field(..., description="Top-k for sampling.")
    top_p: float = Field(..., description="Top-p (nucleus) for sampling.")
    speculative_steps: int = Field(...,
                                   description="Number of speculative steps.")
    value_threshold: float = Field(
        ..., description="Value threshold for accepting speculative generation.")
    max_thought_len: int = Field(...,
                                 description="Maximum length of 'thoughts'.")
    max_retries: int = Field(
        ..., description="Maximum number of retries on failed speculation.")
    max_turns: int = Field(
        10, description="Maximum number of iterations in the agent loop.")
    context_window_size: int = Field(
        2048, description="The number of tokens to retain in history.")
