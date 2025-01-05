import os

from mech_query import run
from typing import Dict, List

# the following is copied over from the mech to simulate the API key retrieval

class KeyChain:
    """Class for managing API keys."""

    def __init__(self, services: Dict[str, List[str]]) -> None:
        """Initialize the KeyChain with a dictionary of service names and corresponding lists of API keys."""
        if not isinstance(services, dict):
            raise ValueError(
                "Services must be a dictionary with service names as keys and lists of API keys as values."
            )

        self.services = services
        self.current_index = {
            service: 0 for service in services
        }  # Start with the first key for each service

    def max_retries(self) -> Dict[str, int]:
        """Get the maximum number of retries for a given service."""
        return {service: len(keys) for service, keys in self.services.items()}

    def rotate(self, service_name: str) -> None:
        """Rotate the current API key for a given service to the next one."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name!r}' not found in KeyChain.")

        # Increment the current index, looping back if at the end of the list
        self.current_index[service_name] += 1
        if self.current_index[service_name] >= len(self.services[service_name]):
            self.current_index[service_name] = 0  # Reset to the start

    def get(self, service_name: str, default_value: str) -> str:
        """Get the current API key for a given service."""
        if service_name not in self.services:
            return default_value

        return self.__getitem__(service_name)

    def __getitem__(self, service_name: str) -> str:
        """Get the current API key for a given service."""
        if service_name not in self.services:
            raise KeyError(f"Service '{service_name!r}' not found in KeyChain.")

        index = self.current_index[service_name]
        return self.services[service_name][index]

import logging
from typing import Any, Callable, Dict, Union


PRICE_NUM_TOKENS = 1000


class TokenCounterCallback:
    """Callback to count the number of tokens used in a generation."""

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
        "gpt-4o-2024-08-06": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "cohere/command-r-plus": {"input": 0.003, "output": 0.015},
        "databricks/dbrx-instruct:nitro": {"input": 0.0009, "output": 0.0009},
        "nousresearch/nous-hermes-2-mixtral-8x7b-sft": {
            "input": 0.00054,
            "output": 0.00054,
        },
    }

    def __init__(self) -> None:
        """Initialize the callback."""
        self.cost_dict: Dict[str, Union[int, float]] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_cost": 0,
            "output_cost": 0,
            "total_cost": 0,
        }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        """Converts a number of tokens to a cost in dollars."""
        return (
            tokens
            / PRICE_NUM_TOKENS
            * TokenCounterCallback.TOKEN_PRICES[model][tokens_type]
        )

    def calculate_cost(
        self, tokens_type: str, model: str, token_counter: Callable, **kwargs: Any
    ) -> None:
        """Calculate the cost of a generation."""
        # Check if it its prompt or tokens are passed in
        prompt_key = f"{tokens_type}_prompt"
        token_key = f"{tokens_type}_tokens"
        if prompt_key in kwargs:
            tokens = token_counter(kwargs[prompt_key], model)
        elif token_key in kwargs:
            tokens = kwargs[token_key]
        else:
            logging.warning(f"No {token_key}_tokens or {tokens_type}_prompt found.")
        cost = self.token_to_cost(tokens, model, tokens_type)
        self.cost_dict[token_key] += tokens
        self.cost_dict[f"{tokens_type}_cost"] += cost

    def __call__(self, model: str, token_counter: Callable, **kwargs: Any) -> None:
        """Callback to count the number of tokens used in a generation."""
        if model not in list(TokenCounterCallback.TOKEN_PRICES.keys()):
            raise ValueError(f"Model {model} not supported.")
        try:
            self.calculate_cost("input", model, token_counter, **kwargs)
            self.calculate_cost("output", model, token_counter, **kwargs)
            self.cost_dict["total_tokens"] = (
                self.cost_dict["input_tokens"] + self.cost_dict["output_tokens"]
            )
            self.cost_dict["total_cost"] = (
                self.cost_dict["input_cost"] + self.cost_dict["output_cost"]
            )
        except Exception as e:
            logging.error(f"Error in TokenCounterCallback: {e}")


keys = KeyChain({
    "coingecko": [os.getenv('COINGECKO_API_KEY')],
    "openai": [os.getenv('OPENAI_API_KEY')],
    "perplexity": [os.getenv('PERPLEXITY_API_KEY')],
    "flipside": [os.getenv('FLIPSIDE_API_KEY')],
})
prompt = "Show transactions for 0x7bfee91193d9df2ac0bfe90191d40f23c773c060 in the last 2 weeks"

print(run(
    prompt=prompt,
    api_keys=keys,
    counter_callback=TokenCounterCallback(),
))
