"""
Main agent script for interacting with the Transformer model using PyTorch.
This script manages the "thought -> tool -> observation" loop,
allowing the model to use tools to complete tasks.
"""
from accelerate import Accelerator

from src.config.core import GenerateConfig
from src.model.factory import load_model_and_tokenizer
from src.utils.decorators import main_entrypoint
from src.utils.setup import setup_logging
from src.utils.setup import setup_from_args



from src.agent.executor import AgentExecutor

@main_entrypoint
def main():
    """Main agent loop for the PyTorch model."""
    setup_logging()
    app_setup = setup_from_args(GenerateConfig)
    accelerator = Accelerator()

    model, tokenizer = load_model_and_tokenizer(
        app_setup.model_name,
        app_setup.config,
        app_setup.args.load_in_4bit,
        app_setup.args.quantized,
    )

    executor = AgentExecutor(
        model=model,
        tokenizer=tokenizer,
        config=app_setup.config,
        accelerator=accelerator,
    )
    executor.run()

if __name__ == "__main__":
    main()
