class MissingEnvVarException(ValueError):
    """A class that defines env var exceptions."""

    def __init__(self, parameter_name: str) -> None:
        """Initialize the exception with the parameter name."""
        self.parameter_name = parameter_name
        self.message = f"Missing required env var: {parameter_name}"
        super().__init__(self.message)


class ToolError(ValueError):
    """A class that defines tool error."""

    def __init__(self, tool: str, message: str) -> None:
        """Initialize the exception with the parameter name."""
        self.message = f"Bad tool {tool}, error: {message}"
        super().__init__(self.message)


class AppError(Exception):
    """A class that app runtime errors."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class ToolNotFoundError(AppError):

    def __init__(self, tool: str):
        super().__init__(f"Tool {tool} not found!", status_code=404)
