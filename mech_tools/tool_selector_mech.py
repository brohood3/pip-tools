import yaml
from typing import Dict, Optional, Tuple, Any, List, Callable
import re
from openai import OpenAI
import functools

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

def with_key_rotation(func: Callable) -> Callable:
    """Decorator to handle API key rotation."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        api_keys = kwargs.get("api_keys", {})
        retries_left = {"openai": len(api_keys.get("openai", [])) - 1}
        
        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except Exception as e:
                if retries_left["openai"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                api_keys.rotate("openai")
                return execute()
            
        return execute()
    return wrapper

class APIClients:
    """Class to manage API clients."""
    def __init__(self, api_keys: Dict[str, List[str]]):
        """Initialize API clients with keys."""
        self.openai_key = api_keys["openai"][0]
        self.client = OpenAI()

def load_tool_config() -> Dict:
    """Load the tool configuration from YAML file."""
    with open('mech_tools.yaml', 'r') as f:
        return yaml.safe_load(f)

def analyze_intent(client: OpenAI, prompt: str) -> str:
    """First stage: Analyze the user's intent."""
    analysis_prompt = f"""Given this user request: "{prompt}"

Analyze the request and identify:
1. What is the user trying to achieve?
2. What type of data or analysis are they looking for?
3. What specific entities are mentioned (tokens, wallets, timeframes)?
4. Is this about past, present, or future information?

Provide your analysis."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant analyzing user requests."},
            {"role": "user", "content": analysis_prompt}
        ]
    )
    return response.choices[0].message.content

def select_tool(client: OpenAI, analysis: str) -> str:
    """Second stage: Select the appropriate tool based on the analysis."""
    tools = load_tool_config()
    tool_descriptions = "\n".join([
        f"- {name}: {tool_data['description']}\n  Example prompts:\n    " + "\n    ".join([f"- {prompt}" for prompt in tool_data.get('example_prompts', [])])
        for name, tool_data in tools['tools'].items()
    ])

    selection_prompt = f"""Based on this analysis: "{analysis}"

Available tools:
{tool_descriptions}

Select the most appropriate tool. Use the exact tool name from the list above.
Pay special attention to the example prompts - if the user's request is similar to an example prompt for a tool, that tool is likely the best choice.

Respond in this format:
TOOL: [exact tool name or "none"]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant selecting the appropriate tool. Always use the exact tool name provided."},
            {"role": "user", "content": selection_prompt}
        ]
    )
    return response.choices[0].message.content

def parse_tool_selection(selection_response: str) -> Tuple[str, str, str]:
    """Parse the tool selection response."""
    tool_match = re.search(r'TOOL:\s*(\w+|none)', selection_response, re.IGNORECASE)
    confidence_match = re.search(r'CONFIDENCE:\s*(\w+)', selection_response, re.IGNORECASE)
    reasoning_match = re.search(r'REASONING:\s*(.+)', selection_response, re.IGNORECASE)
    
    tool = tool_match.group(1).lower() if tool_match else 'none'
    confidence = confidence_match.group(1).lower() if confidence_match else 'low'
    reasoning = reasoning_match.group(1) if reasoning_match else 'No reasoning provided'
    
    return tool, confidence, reasoning

def get_tool_for_prompt(clients: APIClients, prompt: str) -> Dict:
    """Main function to determine which tool to use for a given prompt."""
    tools = load_tool_config()
    
    # Stage 1: Analyze intent
    intent_analysis = analyze_intent(clients.client, prompt)
    
    # Stage 2: Select tool
    tool_selection = select_tool(clients.client, intent_analysis)
    
    # Parse selection
    tool, confidence, reasoning = parse_tool_selection(tool_selection)
    
    # Return appropriate response based on confidence
    if tool != 'none' and confidence in ['high', 'medium']:
        return {
            'tool': tool,
            'prompt': prompt,
            'confidence': confidence,
            'reasoning': reasoning
        }
    else:
        capabilities = [f"- {tool_data['description']}" 
                       for tool_data in tools['tools'].values()]
        return {
            'tool': 'none',
            'message': f"Cannot confidently match this request to available tools. Our tools can help with:\n" + "\n".join(capabilities)
        }

@with_key_rotation
def run(**kwargs) -> MechResponse:
    """Run the tool selector."""
    try:
        # Initialize API clients
        clients = APIClients(kwargs["api_keys"])
        
        # Get the prompt
        prompt = kwargs["prompt"]
        
        # Get tool selection
        result = get_tool_for_prompt(clients, prompt)
        
        # Format the response
        if result['tool'] != 'none':
            response_text = f"Selected tool: {result['tool']}\nPrompt: {result['prompt']}\nConfidence: {result['confidence']}\nReasoning: {result['reasoning']}"
            metadata = {
                'tool': result['tool'],
                'prompt': result['prompt'],
                'confidence': result['confidence'],
                'reasoning': result['reasoning']
            }
        else:
            response_text = result['message']
            metadata = {'error': 'No suitable tool found'}
            
        return response_text, None, metadata, None, kwargs["api_keys"]
        
    except Exception as e:
        return str(e), "", None, None, kwargs["api_keys"] 