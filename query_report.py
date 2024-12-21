from dune_client.client import DuneClient
import os
from openai import OpenAI
from typing import Dict, Any, Optional
import json

def get_dune_results(query_id: int, api_key: str) -> Optional[Dict[Any, Any]]:
    """Fetch the latest results from a Dune query"""
    try:
        dune = DuneClient(api_key)
        result = dune.get_latest_result(query_id)
        
        # Convert ResultsResponse to dictionary
        if hasattr(result, 'result'):
            # Limit to 50 rows and convert to dictionary
            rows = result.result.rows[:50] if len(result.result.rows) > 50 else result.result.rows
            return {'result': rows}
        else:
            print("Invalid or empty response from Dune API")
            return None
            
    except Exception as e:
        print(f"Error fetching Dune results: {e}")
        return None

def generate_summary(data: Dict[Any, Any], openai_api_key: str) -> str:
    """Generate a summary of the Dune query results using GPT"""
    client = OpenAI(api_key=openai_api_key)
    
    try:
        # Extract just the result data if it exists
        result_data = data.get('result', data)
        
        # Convert data to a readable format
        formatted_data = json.dumps(result_data, indent=2)
        
        prompt = f"""Please analyze the following blockchain data and provide a clear, 
        concise summary of the key findings and trends:

        {formatted_data}

        Please include:
        1. Key metrics and their significance
        2. Any notable trends or patterns
        3. Potential implications
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except json.JSONDecodeError as e:
        return f"Error formatting data: {e}"
    except Exception as e:
        return f"Error generating summary: {e}"

def main():
    # Load API keys from environment variables
    dune_api_key = os.getenv("DUNE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not dune_api_key or not openai_api_key:
        print("Please set DUNE_API_KEY and OPENAI_API_KEY environment variables")
        return

    # Get query ID from user
    try:
        query_id = int(input("Enter Dune query ID: ").strip())
    except ValueError:
        print("Please enter a valid number")
        return

    # Fetch Dune results
    results = get_dune_results(query_id, dune_api_key)
    if not results:
        return

    # Generate summary using GPT
    summary = generate_summary(results, openai_api_key)
    
    # Print results
    print("\n=== Query Results ===")
    print(json.dumps(results, indent=2))
    print("\n=== AI Summary ===")
    print(summary)

if __name__ == "__main__":
    main()