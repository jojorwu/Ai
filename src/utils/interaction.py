"""
Utilities for interactive user prompts.
"""
import logging
import os


def select_model_interactively() -> str | None:
    """
    Lists available models in the 'models' directory and prompts the user to select one.
    Returns the selected model name or None if no selection is made.
    """
    models_dir = 'models'
    if not os.path.isdir(models_dir) or not os.listdir(models_dir):
        logging.error("No models found in the '%s' directory.", models_dir)
        return None

    available_models = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    if not available_models:
        logging.error("No valid model directories found in '%s'.", models_dir)
        return None
    if len(available_models) == 1:
        logging.info("Automatically selecting the only available model: %s",
                     available_models[0])
        return available_models[0]

    logging.info("Available models:")
    for i, model_name in enumerate(available_models):
        logging.info("  %d: %s", i + 1, model_name)

    while True:
        try:
            choice = int(input("Please select a model by number: "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            logging.warning("Invalid number. Please try again.")
        except ValueError:
            logging.warning("Invalid input. Please enter a number.")
        except (KeyboardInterrupt, EOFError):
            logging.info("\nSelection cancelled.")
            return None
