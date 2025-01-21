"""
Query Extract Tool

Script for extracting specific information from Dune Analytics queries using natural language questions.
Requires Dune Analytics and OpenAI API keys set in environment variables.
"""

# --- Imports ---
import os
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv
from dune_client.client import DuneClient
from openai import OpenAI
from fastapi import HTTPException


class QueryExtract:
    def __init__(self):
        """Initialize the QueryExtract tool with API clients"""
        load_dotenv()

        # API Keys
        self.dune_api_key = os.getenv("DUNE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.dune_api_key:
            raise HTTPException(
                status_code=500, detail="Missing DUNE_API_KEY environment variable"
            )
        if not self.openai_api_key:
            raise HTTPException(
                status_code=500, detail="Missing OPENAI_API_KEY environment variable"
            )

        # Initialize clients
        self.openai_client = OpenAI()
        self.dune_client = DuneClient(self.dune_api_key)

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for the tool
        
        Args:
            prompt: User's query extraction request
            system_prompt: Optional custom system prompt for the analysis
            
        Returns:
            Dict containing extracted information and metadata
        """
        # Extract query ID and question
        query_id, cleaned_question = self._extract_query_details(prompt)
        if not query_id:
            raise HTTPException(
                status_code=400,
                detail="Could not determine Dune query ID. Please provide a valid query ID.",
            )

        # Get query results
        results = self._get_dune_results(query_id)
        if not results:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch results for query {query_id}. Please verify the query exists and has recent results.",
            )

        # Extract specific information
        answer = self._extract_specific_info(results, prompt, system_prompt)
        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Error extracting information. Please try again.",
            )

        # Return structured response
        return {
            "response": answer,
            "metadata": {
                "query_id": query_id,
                "question": cleaned_question,
                "query_metadata": results["metadata"],
            },
        }

    def _get_dune_results(self, query_id: int) -> Optional[Dict[Any, Any]]:
        """Fetch the latest results from a Dune query"""
        try:
            result = self.dune_client.get_latest_result(query_id)

            if not hasattr(result, "result"):
                return None

            # Store total row count before limiting
            total_rows = len(result.result.rows)

            # Get column names if available
            column_names = (
                list(result.result.rows[0].keys()) if result.result.rows else []
            )

            # Limit to 100 rows by default
            rows = result.result.rows[:100]

            return {
                "result": rows,
                "metadata": {
                    "total_row_count": total_rows,
                    "returned_row_count": len(rows),
                    "column_names": column_names,
                    "execution_time": (
                        result.execution_time
                        if hasattr(result, "execution_time")
                        else None
                    ),
                    "last_refresh_time": (
                        result.last_refresh_time
                        if hasattr(result, "last_refresh_time")
                        else None
                    ),
                },
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error fetching Dune results: {str(e)}"
            )

    def _extract_specific_info(self, data: Dict[Any, Any], question: str, system_prompt: Optional[str] = None) -> str:
        """Extract specific information from query results based on the question"""
        try:
            result_data = data.get("result", [])
            metadata = data.get("metadata", {})

            # Add warning about data limitation if necessary
            data_limitation_note = ""
            if metadata.get("total_row_count", 0) > metadata.get(
                "returned_row_count", 0
            ):
                data_limitation_note = f"""Note: This query contains {metadata.get('total_row_count')} rows in total, 
but only the first {metadata.get('returned_row_count')} rows are shown for analysis."""

            prompt = f"""As a data analyst, extract specific information from this query result to answer the following question:

QUESTION:
{question}

DATA STRUCTURE:
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

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt if system_prompt else "You are a precise data analyst who extracts specific information from query results. Provide direct, focused answers using only the data available.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in analysis: {str(e)}")

    def _extract_query_details(self, prompt: str) -> tuple[Optional[int], str]:
        """Extract query ID and the specific question from the prompt"""
        try:
            # Extract query ID
            id_patterns = [
                r"query (\d+)",
                r"query id (\d+)",
                r"dune (\d+)",
                r"dune query (\d+)",
                r"#(\d+)",
                r"id: (\d+)",
                r"id (\d+)",
            ]

            query_id = None
            matched_pattern = None
            for pattern in id_patterns:
                match = re.search(pattern, prompt.lower())
                if match:
                    query_id = int(match.group(1))
                    matched_pattern = pattern
                    break

            # Extract the question
            question = prompt
            if query_id and matched_pattern:
                question = re.sub(
                    matched_pattern, "", question, flags=re.IGNORECASE
                ).strip()

            # Clean up the question
            question = re.sub(r"\s+", " ", question).strip()
            question = question.strip("?. ")

            return query_id, question

        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error parsing query details: {str(e)}"
            )


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return QueryExtract().run(prompt, system_prompt)
