"""
Fundamental Analysis Tool

A script for comprehensive fundamental analysis of cryptocurrency tokens.
Uses CoinGecko, OpenAI, and Perplexity APIs for data and analysis.
"""

# --- Imports ---
import os
from typing import Dict, List, Optional, TypedDict, Any
from datetime import datetime
import requests
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import HTTPException

# --- Type Definitions ---
class TokenDetails(TypedDict):
    name: str
    symbol: str
    chain: str
    contract_address: str
    description: str
    market_cap: float
    market_cap_fdv_ratio: float
    price_change_24h: float
    price_change_14d: float
    twitter_followers: int
    links: Dict[str, List[str]]

class FundamentalAnalysis:
    def __init__(self):
        """Initialize the Fundamental Analysis tool with required API clients"""
        load_dotenv()
        
        # API Keys
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.perplexity_api_key = os.getenv('PERPLEXITY_API_KEY')
        
        if not self.coingecko_api_key:
            raise HTTPException(status_code=500, detail="Missing COINGECKO_API_KEY environment variable")
        if not self.openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY environment variable")
        if not self.perplexity_api_key:
            raise HTTPException(status_code=500, detail="Missing PERPLEXITY_API_KEY environment variable")
            
        # Initialize API clients
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

    def run(self, prompt: str) -> Dict[str, Any]:
        """Main entry point for the tool"""
        # Get CoinGecko ID
        token_id = self._get_coingecko_id_from_prompt(prompt)
        if not token_id:
            raise HTTPException(
                status_code=400,
                detail="Could not determine the cryptocurrency token from your request. Please provide a clearer token name."
            )
        
        # Get token details
        token_details = self._get_token_details(token_id, prompt)
        if not token_details:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch token details for {token_id}"
            )
        
        # Get analysis
        analysis = self._get_investment_analysis(token_details, prompt)
        if not analysis:
            raise HTTPException(
                status_code=500,
                detail="Error generating analysis"
            )
        
        # Return structured response
        return {
            "token": {
                "id": token_id,
                "name": token_details['name'],
                "symbol": token_details['symbol']
            },
            "metrics": {
                "market_cap": token_details['market_cap'],
                "market_cap_fdv_ratio": token_details['market_cap_fdv_ratio'],
                "price_change_24h": token_details['price_change_24h'],
                "price_change_14d": token_details['price_change_14d'],
                "twitter_followers": token_details['twitter_followers']
            },
            "analysis": analysis
        }

    def _get_token_details(self, token_id: Optional[str], token_name: str) -> Optional[TokenDetails]:
        """Get detailed information about a token from CoinGecko."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            headers = {
                "accept": "application/json",
                "x-cg-demo-api-key": self.coingecko_api_key
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Get first platform as chain and its contract address
            platforms = data.get('platforms', {})
            chain = next(iter(platforms.keys())) if platforms else 'unknown'
            contract_address = platforms.get(chain, '') if platforms else ''
            
            return TokenDetails(
                name=data['name'],
                symbol=data['symbol'].upper(),
                chain=chain,
                contract_address=contract_address,
                description=data.get('description', {}).get('en', ''),
                market_cap=data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                market_cap_fdv_ratio=data.get('market_data', {}).get('market_cap_fdv_ratio', 0),
                price_change_24h=data.get('market_data', {}).get('price_change_percentage_24h', 0),
                price_change_14d=data.get('market_data', {}).get('price_change_percentage_14d', 0),
                twitter_followers=data.get('community_data', {}).get('twitter_followers', 0),
                links=data.get('links', {})
            )
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error fetching token details from CoinGecko: {str(e)}")

    def _get_coingecko_id_from_prompt(self, prompt: str) -> Optional[str]:
        """Use Perplexity to identify the correct CoinGecko ID from a natural language prompt."""
        try:
            perplexity_prompt = f"""Given this analysis request: "{prompt}"

Your task is to identify the exact cryptocurrency and its CoinGecko API ID.

Rules for CoinGecko IDs:
- Must be the official ID used on CoinGecko's website/API
- Always lowercase
- Usually simpler than the token name (e.g., 'bitcoin' not 'bitcoin-btc')
- No special characters except hyphens
- No version numbers or years unless part of official name
- No citations or references
- If you're not 100% certain of the ID, respond with "Cannot determine ID"

Examples:
Input: "Analyze Bitcoin"
Output:
Cryptocurrency: Bitcoin
CoinGecko ID: bitcoin

Input: "Look at Agent AKT fundamentals"
Output:
Cryptocurrency: Akash Network
CoinGecko ID: akash-network

Format your response exactly like the examples above."""

            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{
                    "role": "system",
                    "content": "You are a CoinGecko API expert. Your task is to identify the exact CoinGecko API ID for cryptocurrencies. Be precise and conservative - if unsure about the exact ID, respond with 'Cannot determine ID'."
                }, {
                    "role": "user",
                    "content": perplexity_prompt
                }]
            )
            
            response_text = response.choices[0].message.content
            
            # Check for explicit uncertainty
            if "Cannot determine ID" in response_text:
                return None
            
            coingecko_id = None
            for line in response_text.split('\n'):
                if line.startswith('CoinGecko ID:'):
                    raw_id = line.replace('CoinGecko ID:', '').strip()
                    raw_id = raw_id.split('[')[0].strip('.')
                    coingecko_id = ''.join(c for c in raw_id.lower() if c.isalnum() or c == '-')
                    break
            
            if not coingecko_id:
                return None
                
            # Verify the ID exists
            url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
            headers = {
                "accept": "application/json",
                "x-cg-demo-api-key": self.coingecko_api_key
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return None
                
            return coingecko_id
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error identifying token: {str(e)}")

    def _get_investment_analysis(self, token_details: TokenDetails, original_prompt: str) -> Optional[str]:
        """Get focused tokenomics and market sentiment analysis using GPT."""
        try:
            metrics = []
            if token_details['market_cap'] > 0:
                metrics.append(f"- Market Cap: ${token_details['market_cap']:,.2f}")
            if token_details['market_cap_fdv_ratio'] > 0:
                metrics.append(f"- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}")
            if token_details['price_change_24h'] != 0:
                metrics.append(f"- 24h Price Change: {token_details['price_change_24h']:.2f}%")
            if token_details['price_change_14d'] != 0:
                metrics.append(f"- 14d Price Change: {token_details['price_change_14d']:.2f}%")
            if token_details['twitter_followers'] > 0:
                metrics.append(f"- Social Following: {token_details['twitter_followers']:,} Twitter followers")

            metrics_text = "\n".join(metrics) if metrics else "Market data not available"

            prompt = f"""As an objective tokenomics expert at a top crypto venture capital firm, provide a data-driven analysis of this token:

Original Request: "{original_prompt}"

Token: {token_details['name']} ({token_details['symbol']})
Key Metrics:
{metrics_text}

Focus on:
1. Tokenomics Analysis:
   - Evaluate the Market Cap/FDV ratio and its implications
   - Token distribution patterns and implications
   - Supply dynamics and potential impacts
   - Compare metrics to both successful and failed projects

2. Market Performance:
   - Analyze price action trends and patterns
   - Evaluate social metrics quality and engagement
   - Market conditions impact assessment

Let the data guide your analysis. Highlight both strengths and weaknesses with specific evidence."""

            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data-driven tokenomics researcher at a prestigious crypto venture capital firm. Your analyses are known for being thorough and objective, letting the metrics guide your conclusions."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating analysis: {str(e)}") 