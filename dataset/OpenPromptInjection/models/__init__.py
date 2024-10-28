from .GPT import GPT
from .Llama import Llama
from .Flan import Flan
from .Internlm import Internlm


def create_model(config):
    """
    Factory method to create a LLM instance
    """
    provider = config["model_info"]["provider"].lower()
    if provider == 'gpt':
        model = GPT(config)
    elif provider == 'llama':
        model = Llama(config)
    elif provider == 'flan':
        model = Flan(config)
    elif provider == 'internlm':
        model = Internlm(config)
    else:
        raise ValueError(f"ERROR: Unknown provider {provider}")
    return model