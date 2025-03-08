"""
Brian Transaction API Tool.

This tool leverages the Brian Transaction API to generate blockchain transaction calldata
based on natural language prompts. It supports operations like swapping, bridging,
transferring tokens, depositing/withdrawing from DeFi protocols, and ENS domain registration
across multiple chains.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List, Union
import requests
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)

class BrianTransaction:
    """Class for generating blockchain transactions using Brian API."""

    def __init__(self):
        """Initialize the BrianTransaction class."""
        self.api_endpoint = "https://api.brianknows.org/api/v0/agent/transaction"
        self.api_key = os.environ.get("BRIAN_API_KEY")
        if not self.api_key:
            raise ValueError("Missing BRIAN_API_KEY environment variable")

    def run(self, prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, 
            address: Optional[str] = None, chain_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate blockchain transaction data based on natural language prompt.

        Args:
            prompt: Natural language description of the transaction
            system_prompt: Optional system prompt (not used by this tool)
            model: Optional model specification (not used by this tool)
            address: Address that will send the transaction
            chain_id: Optional chain ID of the user

        Returns:
            Dict containing the response from Brian API
        """
        if not prompt:
            return {"error": "Missing required parameter: prompt", "response": "I need a transaction description to generate data."}
        
        if not address:
            # Try to extract address from prompt, otherwise return an error
            logger.error("Missing required address parameter in BrianTransaction.run")
            return {"error": "Missing required parameter: address", "response": "I need a connected wallet address to generate transaction data."}
        
        # Ensure address is properly formatted (remove any whitespace)
        address = address.strip()
        logger.info(f"Using wallet address: '{address}'")
        
        # Set default chain ID to Ethereum mainnet if not specified
        if not chain_id:
            chain_id = "1"  # Ethereum mainnet
            logger.info(f"No chain_id specified, defaulting to Ethereum mainnet (1)")
        else:
            # Ensure chain_id is properly formatted
            chain_id = chain_id.strip()
            logger.info(f"Using chain_id: '{chain_id}'")
        
        # Modify the prompt to include the wallet address and chain ID
        # This ensures the API can extract these parameters from the prompt text
        chain_name = "Ethereum"  # Default for chain ID 1
        if chain_id == "137":
            chain_name = "Polygon"
        elif chain_id == "56":
            chain_name = "BSC"
        elif chain_id == "42161":
            chain_name = "Arbitrum"
        elif chain_id == "10":
            chain_name = "Optimism"
        elif chain_id == "100":
            chain_name = "Gnosis"
        
        enhanced_prompt = f"{prompt.strip()} using wallet {address} on {chain_name} chain"
        logger.info(f"Enhanced prompt: '{enhanced_prompt}'")
        
        # Prepare the request
        headers = {
            "x-brian-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": enhanced_prompt,
            "address": address,
            "chainId": chain_id
        }
        
        logger.info(f"Calling Brian API with payload: {json.dumps(payload, indent=2)}")
            
        try:
            # Make the API request
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=payload
            )
            
            # Log the response status
            logger.info(f"Brian API response status: {response.status_code}")
            
            # Try to handle error responses with more detail
            if response.status_code != 200:
                error_msg = f"Brian API error: HTTP {response.status_code}"
                try:
                    error_detail = response.json()
                    logger.error(f"Brian API error response: {json.dumps(error_detail, indent=2)}")
                    
                    # Extract more specific error information
                    if 'error' in error_detail:
                        error_msg = f"Brian API error: {error_detail['error']}"
                        
                        # Check for token support issues
                        if "tokens in your request are not supported" in error_detail['error']:
                            # Extract the tokens from extractedParams if available
                            tokens = []
                            if 'extractedParams' in error_detail and len(error_detail['extractedParams']) > 0:
                                params = error_detail['extractedParams'][0]
                                if 'token1' in params:
                                    tokens.append(params['token1'])
                                if 'token2' in params:
                                    tokens.append(params['token2'])
                            
                            tokens_str = ", ".join(tokens) if tokens else "specified tokens"
                            chain_name = "the selected chain"
                            if chain_id == "1":
                                chain_name = "Ethereum mainnet"
                            elif chain_id == "137":
                                chain_name = "Polygon"
                            
                            error_msg = f"The {tokens_str} are not supported on {chain_name}. Please try different tokens or a different chain."
                            return {
                                "error": error_msg,
                                "response": f"I couldn't complete this transaction. {error_msg} You might want to try a different token pair or blockchain network."
                            }
                    
                    if 'message' in error_detail:
                        error_msg = f"Brian API error: {error_detail['message']}"
                except:
                    logger.error(f"Failed to parse error response: {response.text}")
                
                return {"error": error_msg, "response": f"I encountered an error while trying to generate the transaction: {error_msg}"}
            
            response.raise_for_status()
            
            api_response = response.json()
            logger.info(f"Received successful response from Brian API")
            
            # Log the complete response for debugging
            logger.info(f"Complete Brian API response: {json.dumps(api_response, indent=2)}")
            
            # Format the response
            result = {
                "text": self._format_transaction_description(api_response),
                "metadata": api_response,
                "transaction_data": api_response  # Include the complete API response as transaction_data
            }
            
            # Ensure the result has a 'response' field for consistency with other tools
            result["response"] = result["text"]
            
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error calling Brian API: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"Brian API error response: {json.dumps(error_detail, indent=2)}")
                    if 'message' in error_detail:
                        error_msg = f"Brian API error: {error_detail['message']}"
                except:
                    logger.error(f"Failed to parse error response: {e.response.text if hasattr(e.response, 'text') else 'No response text'}")
            
            return {"error": error_msg}
    
    def _format_transaction_description(self, api_response: Dict[str, Any]) -> str:
        """
        Format the API response into a human-readable description.
        
        Args:
            api_response: Raw response from the Brian API
            
        Returns:
            Formatted description of the transaction
        """
        if not api_response or "result" not in api_response or not api_response["result"]:
            return "No transaction data available"
        
        # Get the first result (there should be at least one)
        transaction = api_response["result"][0]
        
        # Extract details
        action = transaction.get("action", "Unknown action")
        tx_type = transaction.get("type", "Unknown type")
        
        # Extract data if available
        data = transaction.get("data", {})
        description = data.get("description", "No description available")
        
        # Get token information
        from_token = data.get("fromToken", {})
        from_amount = data.get("fromAmount", "Unknown amount")
        from_symbol = from_token.get("symbol", "Unknown token")
        
        to_token = data.get("toToken", {})
        to_amount = data.get("toAmount", "Unknown amount")
        to_symbol = to_token.get("symbol", "Unknown token")
        
        # Get chain information
        from_chain_id = data.get("fromChainId", "Unknown chain")
        to_chain_id = data.get("toChainId", from_chain_id)
        
        # Get USD values if available
        from_amount_usd = data.get("fromAmountUSD", "Unknown USD value")
        to_amount_usd = data.get("toAmountUSD", "Unknown USD value")
        gas_cost_usd = data.get("gasCostUSD", "Unknown gas cost")
        
        # Count number of steps
        steps = data.get("steps", [])
        step_count = len(steps)
        
        # Build the description
        if description:
            return description
        
        # If no description provided, build one
        formatted_description = f"Generated {action} transaction ({tx_type}): "
        
        if action.lower() == "swap":
            formatted_description += f"Swap {from_amount} {from_symbol} for approximately {to_amount} {to_symbol}"
            if from_chain_id != "Unknown chain":
                formatted_description += f" on chain ID {from_chain_id}"
        elif action.lower() == "bridge":
            formatted_description += f"Bridge {from_amount} {from_symbol} from chain {from_chain_id} to {to_amount} {to_symbol} on chain {to_chain_id}"
        elif action.lower() == "transfer":
            formatted_description += f"Transfer {from_amount} {from_symbol}"
            if "toAddress" in data:
                formatted_description += f" to {data['toAddress']}"
            if from_chain_id != "Unknown chain":
                formatted_description += f" on chain ID {from_chain_id}"
        else:
            formatted_description += description if description else f"{action} {from_amount} {from_symbol}"
        
        # Add transaction steps info
        formatted_description += f"\nTransaction requires {step_count} step(s)"
        
        # Add estimated gas cost if available
        if gas_cost_usd != "Unknown gas cost":
            formatted_description += f"\nEstimated gas cost: ${gas_cost_usd}"
        
        return formatted_description


def run(prompt: str, system_prompt: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Entry point for the Brian Transaction API tool.
    
    Args:
        prompt: Natural language description of the transaction
        system_prompt: Optional system prompt
        model: Optional model specification
        **kwargs: Additional keyword arguments:
            address: Address that will send the transaction
            chain_id: Optional chain ID of the user
            
    Returns:
        Dict containing the response from Brian API
    """
    try:
        # Extract additional parameters
        address = kwargs.get("address")
        chain_id = kwargs.get("chain_id")
        
        # Add detailed logging
        logger.info(f"Brian Transaction tool called with prompt: '{prompt}'")
        logger.info(f"Address parameter: '{address}'")
        logger.info(f"Chain ID parameter: '{chain_id}'")
        
        if not address:
            logger.error("Missing required address parameter")
            return {"error": "Missing required parameter: address", "response": "I need a connected wallet address to generate transaction data."}
        
        # Initialize and run the tool
        brian_transaction = BrianTransaction()
        result = brian_transaction.run(prompt=prompt, system_prompt=system_prompt, 
                                     model=model, address=address, chain_id=chain_id)
        
        return result
    except Exception as e:
        logger.exception(f"Error running Brian Transaction tool: {str(e)}")
        return {"error": f"Error running Brian Transaction tool: {str(e)}", "response": "There was an error processing your transaction request."}


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python tool.py 'prompt' 'address' [chain_id]")
        sys.exit(1)
    
    example_prompt = sys.argv[1]
    example_address = sys.argv[2]
    example_chain_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = run(example_prompt, address=example_address, chain_id=example_chain_id)
    print(json.dumps(result, indent=2)) 