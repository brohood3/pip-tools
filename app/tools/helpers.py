import importlib
import os
import pkgutil

import yaml

from app import tools
from app.utils.errors import ToolError


def load_module_config(module):
    """Attempts to load the config file from a module."""
    try:
        module_path = os.path.dirname(module.__file__)
        config_path = os.path.join(
            module_path, "config.yaml"
        )  # Adjust if the config file has a different name

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.load(f, Loader=yaml.FullLoader)
                return config_data
        else:
            return None
    except Exception as e:
        return {"error": str(e)}


def load_tools_module(module):
    """Attempts to import the tools.py file from a module."""
    try:
        module_path = os.path.dirname(module.__file__)
        tools_path = os.path.join(module_path, "tool.py")  # Adjust if the file name differs

        if os.path.exists(tools_path):
            module_name = f"{module.__name__}.tool"
            tools_module = importlib.import_module(module_name)
            return tools_module  # Returning the loaded module for inspection
        else:
            return None
    except Exception as e:
        return {"error": str(e)}

# Iterate over all modules in the package
TOOLS = []
TOOL_TO_MODULE = {}
for module_info in pkgutil.iter_modules(tools.__path__):
    module_name = f"{tools.__name__}.{module_info.name}"
    try:
        module = importlib.import_module(module_name)
        config_data = load_module_config(module)
        tools_module = load_tools_module(module)
        if config_data and tools_module:
            if config_data["name"] != module_info.name:
                raise ToolError(
                    module_name,
                    f"Config name '{config_data['name']}' does not match module name '{module_info.name}'",
                )
            TOOLS.append(config_data)
            TOOL_TO_MODULE[module_info.name] = tools_module

    except ImportError as e:
        raise ToolError(module_name, f"Could not load tool: {e}")


if len(TOOLS) != len(TOOL_TO_MODULE):
    raise Exception("Could not load all tools")
