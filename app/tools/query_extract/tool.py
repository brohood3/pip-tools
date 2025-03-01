"""
Query Extract Tool

Extracts specific information from Dune Analytics queries based on user questions.
Provides focused answers using only the available data.
"""

import json
import re
import requests
from typing import Dict, Optional, Any, Tuple, List
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.utils.config import DEFAULT_MODEL
from app.utils.llm import generate_completion

# Load environment variables
load_dotenv()

class QueryExtract:
    """
    A tool for extracting specific information from Dune Analytics queries.
    """

    def __init__(self):
        """Initialize the QueryExtract tool with API keys."""
        self.dune_api_key = os.getenv("DUNE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.dune_api_key:
            raise ValueError("DUNE_API_KEY environment variable is required")
            
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
            
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.dune_base_url = "https://api.dune.com/api/v1"

    def run(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the query extraction process.
        
        Args:
            prompt: The user's question
            system_prompt: Optional system prompt to override default
            model: Optional model to use for extraction
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Extract query ID and question from prompt
            query_id, question = self._extract_query_details(prompt)
            
            if not query_id:
                return {
                    "response": "I couldn't find a valid Dune query ID in your request. Please provide a query ID in the format 'query:12345' or 'dune:12345'.",
                    "metadata": {"query_id": None, "error": True}
                }
                
            # Get query results from Dune
            query_data = self._get_dune_results(query_id)
            
            if not query_data:
                return {
                    "response": f"I couldn't retrieve data for query ID {query_id}. Please check if the query ID is correct and try again.",
                    "metadata": {"query_id": query_id, "error": True}
                }
                
            # Extract specific information based on the question
            answer = self._extract_specific_info(query_data, question, system_prompt, model)
            
            return {
                "response": answer,
                "metadata": {
                    "query_id": query_id,
                    "query_name": query_data.get("metadata", {}).get("query_name", "Unknown"),
                    "error": False
                }
            }
            
        except Exception as e:
            return {
                "response": f"An error occurred: {str(e)}",
                "metadata": {"error": True}
            }

    def _get_dune_results(self, query_id: int) -> Optional[Dict[Any, Any]]:
        """
        Get the results of a Dune query.
        
        Args:
            query_id: The ID of the Dune query
            
        Returns:
            Dictionary with query results or None if failed
        """
        try:
            headers = {
                "x-dune-api-key": self.dune_api_key
            }
            
            # Get the latest execution ID
            execution_url = f"{self.dune_base_url}/query/{query_id}/results"
            response = requests.get(execution_url, headers=headers)
            
            if response.status_code != 200:
                print(f"Failed to get query results: {response.text}")
                return None
                
            return response.json()
            
        except Exception as e:
            print(f"Error getting Dune results: {str(e)}")
            return None

    def _extract_specific_info(self, data: Dict[Any, Any], question: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Extract specific information from query results based on the question.
        
        Args:
            data: The query results data
            question: The user's specific question
            system_prompt: Optional system prompt to override default
            model: Optional model to use for extraction
            
        Returns:
            Extracted answer as a string
        """
        try:
            result_data = data.get("result", {}).get("rows", [])
            metadata = data.get("metadata", {})
            
            # Check if we have data
            if not result_data:
                return "No data was returned from this query."
                
            # Add a note about data limitations if applicable
            data_limitation_note = ""
            if metadata.get("total_row_count", 0) > metadata.get("returned_row_count", 0):
                data_limitation_note = f"NOTE: This query contains {metadata.get('total_row_count')} rows, but only {metadata.get('returned_row_count')} are shown here due to API limitations. The analysis is based only on the available rows."
            
            # Create prompt for the LLM
            prompt = f"""Question: {question}

Query name: {metadata.get('query_name', 'Unknown')}
Available columns: {', '.join(metadata.get('column_names', []))}
Total rows in query: {metadata.get('total_row_count', 0)}
Rows available for analysis: {metadata.get('returned_row_count', 0)}
Last updated: {metadata.get('last_refresh_time', 'Unknown')}

{data_limitation_note}

RAW DATA (Limited to {metadata.get('returned_row_count')} rows):
{json.dumps(result_data, indent=2)}

Please provide:
1. A direct answer to the question using specific numbers/values from the data
2. Only include relevant information that was asked for
3. Format numbers clearly (e.g., percentages, dollar amounts)
4. If the exact information isn't available or might be in the hidden rows, clearly state this limitation"""

            default_system_prompt = "You are a precise data analyst who extracts specific information from query results. Provide direct, focused answers using only the data available."
            
            # Use the LiteLLM utility for completion
            return generate_completion(
                prompt=prompt,
                system_prompt=system_prompt if system_prompt else default_system_prompt,
                model=model,
                temperature=0.3
            )

        except Exception as e:
            return f"Error extracting information: {str(e)}"

    def _extract_query_details(self, prompt: str) -> tuple[Optional[int], str]:
        """
        Extract the query ID and the actual question from the prompt.
        
        Args:
            prompt: The user's input prompt
            
        Returns:
            Tuple of (query_id, question)
        """
        # Look for patterns like query:12345 or dune:12345
        query_patterns = [
            r'(?:query:|dune:)(\d+)',  # Original pattern: query:12345 or dune:12345
            r'(?:query|dune)[:\s]+(\d+)',  # Allow space: query: 12345 or dune 12345
            r'(?:query|dune)(?:\s+id)?[:\s]+(\d+)',  # Allow "id": query id: 12345
            r'(?:id|query|dune)[:\s]+#?(\d+)',  # Allow #: id: #12345
            r'#(\d+)',  # Just a hash: #12345
            r'\b(\d{7,})\b'  # Just a number with 7+ digits (likely a query ID)
        ]
        
        for pattern in query_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                query_id = int(match.group(1))
                
                # Remove the query ID part from the prompt to get the actual question
                # Use the specific pattern that matched to ensure we only remove that part
                question = re.sub(pattern, '', prompt, flags=re.IGNORECASE).strip()
                
                return query_id, question
                
        # If no patterns matched
        return None, prompt


def run(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the query extraction tool.
    
    Args:
        prompt: The user's question
        system_prompt: Optional system prompt to override default
        model: Optional model to use for extraction
        
    Returns:
        Dictionary with response and metadata
    """
    extractor = QueryExtract()
    return extractor.run(prompt, system_prompt, model)
