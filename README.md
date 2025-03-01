# Trading Tools API with LiteLLM Integration

This project provides a collection of trading analysis tools via a FastAPI backend, with flexible model configuration using LiteLLM.

## Features

- **Multiple Trading Tools**: Technical analysis, fundamental analysis, price prediction, and more
- **Flexible Model Configuration**: Use any supported LLM provider through LiteLLM
- **Standardized API**: Consistent interface for all tools

## Model Configuration

The application uses LiteLLM to provide a unified interface for working with different LLM providers. This allows you to:

1. Use different models without changing your code
2. Switch between providers (OpenAI, Anthropic, Google, etc.)
3. Configure models at runtime

### How to Configure Models

Models can be configured in three ways (in order of precedence):

1. **Request-level**: Specify the model in the API request
   ```json
   {
     "prompt": "Your prompt here",
     "system_prompt": "Optional system prompt",
     "model": "claude-3-opus"
   }
   ```

2. **Environment Variable**: Set the `DEFAULT_MODEL` environment variable
   ```
   DEFAULT_MODEL=gpt-4o-mini
   ```

3. **Default Configuration**: Falls back to "gemini-2.0-flash" if not specified

### Supported Models

The following models are pre-configured in the application:

- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- **Anthropic**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Google**: gemini-pro, gemini-1.5-pro
- **Mistral**: mistral-large, mistral-medium, mixtral-8x7b
- **Meta**: llama-3-70b, llama-3-8b

You can add more models by updating the `LITELLM_PROVIDER_MAP` in `app/utils/config.py`.

## API Usage

### List Available Tools

```
GET /tools
```

### Run a Tool

```
POST /{tool_name}

{
  "prompt": "Your prompt here",
  "system_prompt": "Optional system prompt",
  "model": "Optional model name"
}
```

### Tool Selector

```
POST /tool_selector

{
  "prompt": "Your prompt here",
  "system_prompt": "Optional system prompt",
  "model": "Optional model name",
  "allowed_tools": ["tool1", "tool2"]
}
```

## Environment Variables

Required environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI models)
- `GEMINI_API_KEY`: Your Google API key (for Gemini models)
- Other provider keys as needed (ANTHROPIC_API_KEY, etc.)

Optional environment variables:

- `DEFAULT_MODEL`: Default model to use

## Development

### Installation

This project uses Poetry for dependency management. To install dependencies:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Alternatively, if you don't have Poetry
pip install -r requirements.txt
```

### Running the API

```bash
# With Poetry
poetry run uvicorn app.main:app --reload

# Without Poetry
uvicorn app.main:app --reload
```

### Adding a New Tool

1. Create a new directory in `app/tools/`
2. Implement the tool with a `run()` function that accepts `prompt`, `system_prompt`, and `model` parameters
3. Update `app/tools/helpers.py` to register your new tool 