from dune_client.client import DuneClient
import os
from openai import OpenAI
from typing import Dict, Any, Optional, Tuple
import json
import argparse

def get_dune_results(query_id: int, api_key: str) -> Optional[Dict[Any, Any]]:
    """Fetch the latest results from a Dune query"""
    try:
        dune = DuneClient(api_key)
        result = dune.get_latest_result(query_id)
        
        if hasattr(result, 'result'):
            rows = result.result.rows[:50] if len(result.result.rows) > 50 else result.result.rows
            return {'result': rows}
        return None
            
    except Exception as e:
        return None

def generate_summary(data: Dict[Any, Any], openai_api_key: str, query_description: str) -> str:
    """Generate a summary of the Dune query results using GPT"""
    client = OpenAI(api_key=openai_api_key)
    
    try:
        result_data = data.get('result', data)
        formatted_data = json.dumps(result_data, indent=2)
        
        prompt = f"""Please analyze the following blockchain data from a query that {query_description}
        and provide a clear, concise summary of the key findings and trends:

        {formatted_data}

        Please include:
        1. Key metrics and their significance
        2. Any notable trends or patterns
        3. Potential implications
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {e}"

def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """Run the Dune query report tool"""
    try:
        # Extract parameters from kwargs
        query_id = int(kwargs["query_id"])
        query_description = kwargs["query_description"]
        dune_api_key = os.getenv("DUNE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not dune_api_key or not openai_api_key:
            return "Error: Please set DUNE_API_KEY and OPENAI_API_KEY environment variables", None, None, None

        # Get Dune results
        results = get_dune_results(query_id, dune_api_key)
        if not results:
            return "Failed to fetch Dune results", None, None, None

        # Generate summary
        summary = generate_summary(results, openai_api_key, query_description)
        
        # Prepare response
        response = {
            "summary": summary,
            "raw_data": results
        }
        
        # Return tuple of (result, prompt, metadata, callback)
        return json.dumps(response), query_description, results, None

    except KeyError as e:
        return f"Missing required parameter: {e}", None, None, None
    except ValueError as e:
        return f"Invalid parameter value: {e}", None, None, None
    except Exception as e:
        return f"Unexpected error: {e}", None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dune Query Report Tool')
    parser.add_argument('query_id', type=int, help='The Dune query ID')
    parser.add_argument('description', type=str, help='Description of what the query analyzes')
    
    args = parser.parse_args()
    
    result, description, metadata, callback = run(
        query_id=args.query_id,
        query_description=args.description
    )
    
    print("\n=== Results ===")
    print("\nSummary and Data:")
    print(result)