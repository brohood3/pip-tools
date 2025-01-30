import os

from app.utils.errors import MissingEnvVarException
from dotenv import load_dotenv


load_dotenv()


def ensure_var(var_name: str, default: str = None, is_required: bool = True) -> str:
    """Ensure that an env var exists. If it doesn't, return the default value.

    Args:
        var_name (str): Name of the environment variable.
        is_required (bool, optional): Whether the environment variable is required. Defaults to True.
        default (str, optional): Default value if the environment variable is not set. Defaults to None.

    Raises:
        MissingEnvVarException: Raised if the environment variable is not set and is_required is True.

    Returns:
        str: The value of the environment variable.
    """

    if var_name in os.environ:
        return os.environ[var_name]

    if is_required and not default:
        raise MissingEnvVarException(var_name)

    return default


OPENAI_API_KEY = ensure_var("OPENAI_API_KEY")
