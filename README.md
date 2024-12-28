# Converting a Script to a Mech Tool

This guide provides step-by-step instructions for converting a Python script into a mech tool, based on a real-world example of converting `fundamental_analysis.py`.

## Prerequisites

- Your original Python script
- Understanding of the script's dependencies and API requirements

## Steps

### 1. Define the Response Type

Add the standard mech response type at the top of your file:

```python
from typing import List, Dict, Optional, TypedDict, Tuple, Any, Callable

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]
```

### 2. Modify API Client Management

1. Update your API client class to accept keys through parameters instead of environment variables:

```python
class APIClients:
    def __init__(self, api_keys: Any):
        self.service1_key = api_keys["service1"]
        self.service2_key = api_keys["service2"]
        
        if not all([self.service1_key, self.service2_key]):
            raise ValueError("Missing required API keys")
```

### 3. Convert Print Statements to Return Values

Instead of printing results, store them in variables to return:

```python
# Before
print("\nTokenomics & Market Analysis:")
print(tokenomics_analysis)

# After
response = tokenomics_analysis
```

### 4. Create the Run Function

Replace your `main()` function with a `run()` function that accepts kwargs:

```python
@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    try:
        clients = APIClients(api_keys)
        # Your implementation here
        return response, "", None, None
    except Exception as e:
        return str(e), "", None, None
```

### 5. Add Key Rotation Support

Add the key rotation decorator to handle API rate limits:

```python
def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper
```

### 6. Testing

Create or modify the test file (`tool_test.py`):

```python
import os
from your_tool import run
from typing import Dict, List

# Setup keys
keys = KeyChain({
    "service1": [os.getenv('SERVICE1_API_KEY')],
    "service2": [os.getenv('SERVICE2_API_KEY')],
})

# Test the tool
print(run(
    prompt="Your test prompt",
    api_keys=keys,
    counter_callback=TokenCounterCallback(),
))
```

## Best Practices

1. **API Key Management**
   - Remove all direct environment variable access
   - Use the KeyChain class for key management
   - Support key rotation for rate-limited services

2. **Error Handling**
   - Return errors in the mech response format
   - Handle API-specific errors (like rate limits) appropriately
   - Include meaningful error messages

3. **Response Format**
   - Convert all print statements to return values
   - Return the 4-tuple mech response format
   - Use the response string for primary output

4. **Testing**
   - Use the standard test structure
   - Test with real API keys
   - Test error cases and rate limit handling

## Common Issues

1. **Environment Variables**
   - Remove all direct `os.getenv()` calls
   - Pass all API keys through the `api_keys` parameter

2. **Print Statements**
   - Convert all print statements to string responses
   - Combine multiple outputs into a single response string

3. **Main Function**
   - Remove the `if __name__ == "__main__":` block
   - Convert the main logic into the `run()` function
   - Ensure all parameters are passed through kwargs 