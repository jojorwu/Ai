"""
Custom exception hierarchy for the Titans Transformer and Agent system.
"""

class TitansError(Exception):
    """Base class for all exceptions in the Titans project."""
    pass

class ModelConfigError(TitansError, ValueError):
    """Raised when there is an invalid model configuration."""
    pass

class InitializationError(TitansError, RuntimeError):
    """Raised when model initialization fails."""
    pass

class GenerationError(TitansError, RuntimeError):
    """Raised during text generation errors."""
    pass

class AgentExecutionError(TitansError, RuntimeError):
    """Raised when an agent fails to execute a task or tool."""
    pass

class MultimodalError(TitansError, RuntimeError):
    """Raised for errors related to vision or multimodal processing."""
    pass
