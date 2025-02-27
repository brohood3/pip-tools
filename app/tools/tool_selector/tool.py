"""
Tool Selector

A script that analyzes user requests and intelligently selects the appropriate tool from app/tools.
Uses GPT-4 to analyze intent and match requests to available tools based on their configurations.
"""

# --- Imports ---
import yaml
import os
from typing import Dict, Optional, List, TypedDict, Literal, Union, Any
from openai import OpenAI
import re
from dotenv import load_dotenv
from datetime import datetime


# --- Type Definitions ---
class ToolConfig(TypedDict):
    """Type definition for a tool's configuration."""

    name: str
    description: str
    example_prompts: List[str]
    required_params: Dict[str, Dict[str, str]]


class ToolsConfig(TypedDict):
    """Type definition for the complete tools configuration."""

    tools: Dict[str, ToolConfig]


class ToolResult(TypedDict, total=False):
    """Type definition for the tool selection result."""

    tool: str
    prompt: str
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    required_params: Dict[str, Dict[str, str]]
    additional_info: str  # Optional field for notes about multiple tokens/tools


class NoToolResult(TypedDict):
    """Type definition for when no tool is selected."""

    tool: Literal["none"]
    message: str


# --- Configuration Loading ---
def load_tool_configs(allowed_tools: Optional[List[str]] = None) -> ToolsConfig:
    """
    Load tool configurations from app/tools directory.
    
    Args:
        allowed_tools: Optional list of tool names to load. If None, loads all tools.

    Returns:
        Dict containing tool configurations, keyed by tool name.
    """
    tools = {}

    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (app/tools)
    tools_dir = os.path.dirname(current_dir)

    # Iterate through each tool directory
    for tool_dir in os.listdir(tools_dir):
        config_path = os.path.join(tools_dir, tool_dir, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                # Only include tool if it's in allowed_tools (or if allowed_tools is None)
                if allowed_tools is None or config["name"] in allowed_tools:
                    tools[config["name"]] = config

    return {"tools": tools}


# --- Intent Analysis ---
def analyze_intent(client: OpenAI, prompt: str) -> str:
    """
    First stage: Analyze the user's intent using GPT-4.

    Args:
        client: OpenAI client instance
        prompt: User's original request

    Returns:
        Detailed analysis of user's intent and requirements
    """
    analysis_prompt = f"""Given this user request: "{prompt}"

Analyze the request and identify:
1. What is the user trying to achieve? (Include type of analysis/data needed)
2. What specific entities are mentioned (tokens, wallets, timeframes)?
3. Is this about past, present, or future information?
4. Is the user asking for information about how to use a tool (documentation/help) or are they trying to actually use a tool? Be explicit about this distinction.


Provide your analysis."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant analyzing user requests.",
            },
            {"role": "user", "content": analysis_prompt},
        ],
    )
    return response.choices[0].message.content


# --- Tool Selection ---
def select_tool(client: OpenAI, analysis: str, tools_config: ToolsConfig, system_prompt: Optional[str] = None) -> str:
    """
    Second stage: Select the appropriate tool based on the analysis.

    Args:
        client: OpenAI client instance
        analysis: Intent analysis from first stage
        tools_config: Configuration of all available tools
        system_prompt: Optional custom system prompt for tool selection

    Returns:
        Structured response with tool selection and confidence
    """
    # Remove tool_selector from available tools
    available_tools = {
        name: data
        for name, data in tools_config["tools"].items()
        if name != "tool_selector"
    }

    # Build tool descriptions
    tool_descriptions = "\n".join(
        [
            f"- {name}: {tool_data['description']}\n  Example prompts:\n    "
            + "\n    ".join(
                [f"- {prompt}" for prompt in tool_data.get("example_prompts", [])]
            )
            for name, tool_data in available_tools.items()
        ]
    )

    selection_prompt = f"""Based on this analysis: "{analysis}"

Available tools:
{tool_descriptions}

Consider the following when making your selection:
1. If multiple tokens are mentioned, indicate this in your reasoning
2. If the request could be handled by multiple tools, explain the tradeoffs
3. Lower your confidence if:
   - Multiple tokens need to be analyzed separately
   - Multiple tools could be applicable
   - The request is ambiguous between technical and fundamental analysis
4. If the user is asking how to use a tool or requesting information about tool functionality, select "none".

Select the most appropriate tool. Use the exact tool name from the list above.
Pay special attention to the example prompts - if the user's request is similar to an example prompt for a tool, that tool is likely the best choice.

Respond in this format:
TOOL: [exact tool name or "none"]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation]
ADDITIONAL_INFO: [any notes about multiple tokens or tools that might be needed]"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": system_prompt if system_prompt else "You are a helpful assistant selecting the appropriate tool. Always use the exact tool name provided. Be conservative with confidence levels when requests are ambiguous or require multiple tools/tokens.",
            },
            {"role": "user", "content": selection_prompt},
        ],
    )
    return response.choices[0].message.content


def parse_tool_selection(
    selection_response: str,
) -> tuple[str, str, str, Optional[str]]:
    """
    Parse the structured tool selection response.

    Args:
        selection_response: Raw response from GPT-4

    Returns:
        Tuple of (tool_name, confidence_level, reasoning, additional_info)
    """
    tool_match = re.search(r"TOOL:\s*(\w+|none)", selection_response, re.IGNORECASE)
    confidence_match = re.search(
        r"CONFIDENCE:\s*(\w+)", selection_response, re.IGNORECASE
    )
    reasoning_match = re.search(
        r"REASONING:\s*(.+?)(?=\n\w+:|$)", selection_response, re.IGNORECASE | re.DOTALL
    )
    additional_info_match = re.search(
        r"ADDITIONAL_INFO:\s*(.+)", selection_response, re.IGNORECASE | re.DOTALL
    )

    tool = tool_match.group(1).lower() if tool_match else "none"
    confidence = confidence_match.group(1).lower() if confidence_match else "low"
    reasoning = (
        reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
    )
    additional_info = (
        additional_info_match.group(1).strip() if additional_info_match else None
    )

    return tool, confidence, reasoning, additional_info


# --- Main Logic ---
def get_tool_for_prompt(
    client: OpenAI, 
    prompt: str, 
    system_prompt: Optional[str] = None,
    allowed_tools: Optional[List[str]] = None
) -> Union[ToolResult, NoToolResult]:
    """
    Main function to determine which tool to use for a given prompt.

    Args:
        client: OpenAI client instance
        prompt: User's original request
        system_prompt: Optional custom system prompt for tool selection
        allowed_tools: Optional list of allowed tool names. If None, all tools are available.

    Returns:
        Dictionary containing either:
        - Selected tool info with confidence and required parameters
        - No tool selected with explanation message
    """
    tools_config = load_tool_configs(allowed_tools)

    # If no tools are available after filtering
    if not tools_config["tools"]:
        return {
            "response": {
                "tool": "none",
                "message": "No tools are available for this request."
            },
            "metadata": {"prompt": prompt, "timestamp": datetime.now().isoformat()}
        }

    # Stage 1: Analyze intent
    intent_analysis = analyze_intent(client, prompt)
    print("\nINTENT ANALYSIS:")
    print("-" * 50)
    print(intent_analysis)

    # Stage 2: Select tool
    tool_selection = select_tool(client, intent_analysis, tools_config, system_prompt)
    print("\nTOOL SELECTION:")
    print("-" * 50)
    print(tool_selection)

    # Parse selection
    tool, confidence, reasoning, additional_info = parse_tool_selection(tool_selection)

    # Return appropriate response based on confidence
    if tool != "none" and confidence in ["high", "medium"]:
        result = {
            "response": {
                "tool": tool,
                "confidence": confidence,
                "reasoning": reasoning,
                "required_params": tools_config["tools"][tool].get("required_params", {}),
            },
            "metadata": {"prompt": prompt, "timestamp": datetime.now().isoformat()}
        }
        if additional_info:
            result["response"]["additional_info"] = additional_info
        return result
    else:
        capabilities = [
            f"- {tool_data['description']}"
            for name, tool_data in tools_config["tools"].items()
            if name != "tool_selector"
        ]
        return {
            "response": {
                "tool": "none",
                "message": f"Cannot confidently match this request to available tools. Available tools can help with:\n"
                + "\n".join(capabilities),
            },
            "metadata": {"prompt": prompt, "timestamp": datetime.now().isoformat()}
        }


# --- Main Execution ---
def main():
    """Main function to run the tool selector."""
    try:
        # Load environment variables
        load_dotenv()

        # Initialize OpenAI client
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        client = OpenAI()

        # Get input from command line or user input
        import sys

        if len(sys.argv) > 1:
            prompt = " ".join(sys.argv[1:])
        else:
            prompt = input("\nEnter your request: ")

        print(f"\nAnalyzing request: {prompt}")

        # Get tool selection
        result = get_tool_for_prompt(client, prompt)

        # Print results
        print("\nRESULTS:")
        print("-" * 50)
        if result["response"]["tool"] != "none":
            print(f"Selected tool: {result['response']['tool']}")
            print(f"Confidence: {result['response']['confidence']}")
            print(f"Reasoning: {result['response']['reasoning']}")
            print("\nRequired parameters:")
            for param, details in result["response"]["required_params"].items():
                print(f"- {param}: {details}")
        else:
            print(result["response"]["message"])

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

openai_client = OpenAI()


# Add the run function to match the API
def run(prompt: str, system_prompt: Optional[str] = None, allowed_tools: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main entry point for the tool selector.
    
    Args:
        prompt: User's request
        system_prompt: Optional custom system prompt
        allowed_tools: Optional list of allowed tool names
        
    Returns:
        Tool selection result
    """
    return get_tool_for_prompt(openai_client, prompt, system_prompt, allowed_tools)
