#!/usr/bin/env python3
"""
Test script for Trading Tools API.
Tests all tools with multiple prompts and outputs first ~150 words of each response.
Saves results to both JSON and text report files.
"""

import os
import sys
import json
import argparse
import requests
import datetime
import textwrap
from pprint import pformat
from typing import Dict, List, Any, Optional, Tuple

# Constants
API_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "gemini-2.0-flash"
REPORT_DIR = "test_reports"

# Tool definitions with multiple test prompts for each
TOOLS = {
    "ten_word_ta": {
        "description": "Ten-word technical analysis",
        "prompts": [
            "What do you think about BTC price action in the next week?"
        ]
    },
    "general_predictor": {
        "description": "General market prediction",
        "prompts": [
            "Will the S&P 500 rise or fall tomorrow?"
        ]
    },
    "tool_selector": {
        "description": "Tool selection based on user intent",
        "prompts": [
            "I want to understand BTC's price movements",
            "Help me analyze the long term prospects for ETH",
            "What trading tools should I use to assess market sentiment?"
        ]
    },
    "technical_analysis": {
        "description": "Comprehensive technical analysis",
        "prompts": [
            "Perform technical analysis on SOL with focus on support and resistance"
        ]
    },
    "macro_outlook_analyzer": {
        "description": "Macro economic outlook analysis",
        "prompts": [
            "Analyze how recent Fed policies might impact the crypto market"
        ]
    },
    "fundamental_analysis": {
        "description": "Fundamental token analysis",
        "prompts": [
            "Analyze the fundamental value of Ethereum"
        ]
    },
    "lunar_crush_screener": {
        "description": "Screening for promising cryptocurrencies",
        "prompts": [
            "Find cryptocurrencies with high social engagement and low marketcap"
        ]
    },
    "query_extract": {
        "description": "Extracting information from Dune queries",
        "prompts": [
            "Extract insights from Dune query 4508132"
        ]
    },
    "price_predictor": {
        "description": "Detailed price prediction with scenarios",
        "prompts": [
            "Will SOL price go up or down in the next two weeks?"
        ]
    }
}

def get_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Trading Tools API")
    parser.add_argument(
        "--tools", 
        nargs="+", 
        choices=list(TOOLS.keys()),
        help="Specific tools to test (default: all tools)"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=[DEFAULT_MODEL],
        help=f"Models to test with (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Show full responses instead of truncated versions"
    )
    return parser.parse_args()

def truncate_text(text: str, max_words: int = 150) -> str:
    """Truncate text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + "..."

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"

def test_tool(
    tool_name: str, 
    model: str, 
    prompt: str,
    verbose: bool = False
) -> Tuple[Dict[str, Any], float]:
    """Test a specific tool with a specific model and prompt."""
    start_time = datetime.datetime.now()
    
    url = f"{API_BASE_URL}/{tool_name}"
    
    payload = {
        "prompt": prompt,
        "model": model
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Process response for display
        if not verbose and "response" in data:
            data["response"] = truncate_text(data["response"])
            
        return data, duration
    except requests.exceptions.RequestException as e:
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        error_data = {
            "error": str(e),
            "status": "failed"
        }
        return error_data, duration

def get_timestamp() -> str:
    """Get current timestamp in a filename-friendly format."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_json_report(results: Dict[str, Any], timestamp: str) -> str:
    """Save results to a JSON file."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    filename = f"{REPORT_DIR}/test_report_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def save_text_report(results: Dict[str, Any], timestamp: str) -> str:
    """Save results to a formatted text file."""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    
    filename = f"{REPORT_DIR}/test_report_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Trading Tools API Test Report\n")
        f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'=' * 50}\n\n")
        
        # Write summary
        success_count = sum(1 for tool in results["results"].values() 
                          for model_results in tool.values() 
                          for result in model_results 
                          if result["status"] == "success")
        
        failure_count = sum(1 for tool in results["results"].values() 
                          for model_results in tool.values() 
                          for result in model_results 
                          if result["status"] == "failed")
        
        total_count = success_count + failure_count
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        f.write(f"SUMMARY:\n")
        f.write(f"Tools tested: {len(results['results'])}\n")
        f.write(f"Models tested: {len(results['models'])}\n")
        f.write(f"Total tests: {total_count}\n")
        f.write(f"Successful: {success_count} ({success_rate:.1f}%)\n")
        f.write(f"Failed: {failure_count}\n")
        f.write(f"Average response time: {results['average_response_time']:.2f} seconds\n\n")
        
        # Write detailed results
        for tool_name, tool_results in results["results"].items():
            f.write(f"TOOL: {tool_name} - {TOOLS[tool_name]['description']}\n")
            f.write(f"{'-' * 50}\n\n")
            
            for model, prompts_results in tool_results.items():
                f.write(f"  MODEL: {model}\n")
                
                for i, result in enumerate(prompts_results):
                    prompt = result.get("prompt", "Unknown prompt")
                    status = result.get("status", "unknown")
                    duration = result.get("duration", 0)
                    
                    f.write(f"    PROMPT {i+1}: {prompt}\n")
                    f.write(f"    STATUS: {status.upper()}\n")
                    f.write(f"    TIME: {format_duration(duration)}\n")
                    
                    if status == "success" and "response" in result:
                        response = result["response"]
                        # Format the response with proper indentation
                        wrapped_response = textwrap.fill(response, width=70, 
                                                       initial_indent="      ", 
                                                       subsequent_indent="      ")
                        f.write(f"    RESPONSE PREVIEW:\n{wrapped_response}\n")
                    elif status == "failed" and "error" in result:
                        f.write(f"    ERROR: {result['error']}\n")
                        
                    f.write("\n")
                
                f.write("\n")
            
            f.write("\n")
            
    return filename

def main():
    """Main test execution function."""
    args = get_args()
    
    # Determine which tools to test
    tool_names = args.tools if args.tools else list(TOOLS.keys())
    models = args.models
    verbose = args.verbose
    
    print(f"Testing Trading Tools API")
    print(f"========================\n")
    print(f"Tools to test: {', '.join(tool_names)}")
    print(f"Models to test: {', '.join(models)}")
    print(f"Verbose mode: {'On' if verbose else 'Off'}")
    print(f"Results will be saved to the '{REPORT_DIR}' directory\n")
    
    # Initialize results dictionary
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tools": tool_names,
        "models": models,
        "results": {},
        "total_tests": 0,
        "successful_tests": 0,
        "failed_tests": 0,
        "total_response_time": 0,
        "average_response_time": 0
    }
    
    total_tests = 0
    successful_tests = 0
    failed_tests = 0
    total_response_time = 0
    
    try:
        # Test each tool with each model
        for tool_name in tool_names:
            print(f"\nTesting {tool_name} ({TOOLS[tool_name]['description']}):")
            results["results"][tool_name] = {}
            
            for model in models:
                print(f"  With model {model}:")
                results["results"][tool_name][model] = []
                
                for i, prompt in enumerate(TOOLS[tool_name]["prompts"]):
                    print(f"    Prompt {i+1}: ", end="", flush=True)
                    
                    response_data, duration = test_tool(tool_name, model, prompt, verbose)
                    
                    # Store results
                    test_result = {
                        "prompt": prompt,
                        "duration": duration,
                        **response_data
                    }
                    
                    if "error" in response_data:
                        test_result["status"] = "failed"
                        print(f"FAILED ({format_duration(duration)})")
                        print(f"      Error: {response_data['error']}")
                        failed_tests += 1
                    else:
                        test_result["status"] = "success"
                        print(f"SUCCESS ({format_duration(duration)})")
                        print(f"      Preview: {truncate_text(response_data.get('response', ''), 30)}")
                        successful_tests += 1
                    
                    total_tests += 1
                    total_response_time += duration
                    
                    results["results"][tool_name][model].append(test_result)
    
        # Calculate summary statistics
        results["total_tests"] = total_tests
        results["successful_tests"] = successful_tests
        results["failed_tests"] = failed_tests
        results["total_response_time"] = total_response_time
        results["average_response_time"] = total_response_time / total_tests if total_tests > 0 else 0
        
        # Save results to files
        timestamp = get_timestamp()
        json_filename = save_json_report(results, timestamp)
        text_filename = save_text_report(results, timestamp)
        
        # Print summary
        print("\nTest Summary:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {successful_tests} ({(successful_tests/total_tests*100):.1f}%)")
        print(f"  Failed: {failed_tests}")
        print(f"  Average response time: {format_duration(results['average_response_time'])}")
        print(f"\nReports saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  Text: {text_filename}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 