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
from app.utils.config import DEFAULT_MODEL


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
    links: Dict[str, Any]


class FundamentalAnalysis:
    def __init__(self):
        """Initialize OpenAI client and API keys"""
        self.openai_client = OpenAI()
        self.perplexity_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai",
        )
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")

    def get_token_details(self, token_id: str) -> Optional[TokenDetails]:
        """Get detailed information about a token from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            headers = {
                "accept": "application/json",
                "x-cg-demo-api-key": self.coingecko_api_key,
            }

            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            platforms = data.get("platforms", {})
            chain = next(iter(platforms.keys())) if platforms else "ethereum"
            contract_address = platforms.get(chain, "") if platforms else ""

            return TokenDetails(
                name=data["name"],
                symbol=data["symbol"].upper(),
                chain=chain,
                contract_address=contract_address,
                description=data.get("description", {}).get("en", ""),
                market_cap=data.get("market_data", {})
                .get("market_cap", {})
                .get("usd", 0),
                market_cap_fdv_ratio=data.get("market_data", {}).get(
                    "market_cap_fdv_ratio", 0
                ),
                price_change_24h=data.get("market_data", {}).get(
                    "price_change_percentage_24h", 0
                ),
                price_change_14d=data.get("market_data", {}).get(
                    "price_change_percentage_14d", 0
                ),
                twitter_followers=data.get("community_data", {}).get(
                    "twitter_followers", 0
                ),
                links=data.get("links", {}),
            )

        except requests.exceptions.RequestException as e:
            print(f"Error fetching token details: {e}")
            return None

    def get_investment_analysis(
        self, token_details: Optional[TokenDetails], original_prompt: str
    ) -> Optional[str]:
        """Generate tokenomics and investment analysis using OpenAI."""
        if not token_details:
            return None

        try:
            prompt = f"""Analyze the tokenomics and investment potential for {token_details['name']} ({token_details['symbol']}) based on the following data:

Market Cap: ${token_details['market_cap']:,.2f}
Market Cap / FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
24h Price Change: {token_details['price_change_24h']:.2f}%
14d Price Change: {token_details['price_change_14d']:.2f}%
Twitter Followers: {token_details['twitter_followers']:,}

Focus on:
1. Tokenomics analysis (supply, distribution, inflation)
2. Market positioning and competitive advantages
3. Investment thesis (bull and bear cases)
4. Key metrics to monitor
5. Risk assessment

Original user query: "{original_prompt}"

Provide a detailed, balanced analysis that considers both positive and negative factors."""

            system_prompt = "You are a cryptocurrency investment analyst specializing in tokenomics and fundamental analysis. Provide balanced, data-driven insights that consider both bull and bear cases."
            
            # Use the LiteLLM utility instead of direct OpenAI call
            from app.utils.llm import generate_completion
            
            return generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                model=DEFAULT_MODEL,
                temperature=0.7
            )

        except Exception as e:
            print(f"Error generating investment analysis: {e}")
            return None

    def get_project_research(
        self, token_details: Optional[TokenDetails], original_prompt: str
    ) -> Optional[str]:
        """Research the project using Perplexity API"""
        try:
            # Base prompt with original user request
            base_prompt = f"""Original Request: "{original_prompt}"

"""
            # Add project data if available
            if token_details:
                research_links = []
                important_link_types = [
                    "homepage",
                    "blockchain_site",
                    "whitepaper",
                    "announcement_url",
                    "twitter_screen_name",
                    "telegram_channel_identifier",
                    "github_url",
                ]

                for link_type, urls in token_details["links"].items():
                    if link_type in important_link_types:
                        if isinstance(urls, list):
                            research_links.extend([url for url in urls if url])
                        elif isinstance(urls, str) and urls:
                            if link_type == "telegram_channel_identifier":
                                research_links.append(f"https://t.me/{urls}")
                            else:
                                research_links.append(urls)

                links_text = "\n".join([f"- {url}" for url in research_links])

                base_prompt += f"""Project: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Contract: {token_details['contract_address']}
Description: {token_details['description']}

Available Sources:
{links_text}
"""

            prompt = (
                base_prompt
                + """As the lead blockchain researcher at a top-tier crypto investment fund, conduct comprehensive due diligence for our portfolio managers.

Your research will be used by:
- Portfolio managers making 7-8 figure allocation decisions
- Risk assessment teams evaluating project viability
- Investment committee members reviewing opportunities

Please provide an institutional-grade analysis covering:
1. Project Overview & Niche:
   - What problem does it solve?
   - What's unique about their approach?
   - What is their competition?

2. Ecosystem Analysis:
   - Key partnerships and integrations
   - Developer activity and community
   - Infrastructure and technology stack

3. Recent & Upcoming Events:
   - Latest developments
   - Roadmap milestones
   - Upcoming features or releases"""
            )

            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior blockchain researcher at a $500M crypto fund. Your research directly influences investment allocation decisions. Maintain professional skepticism and support claims with evidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating project research: {e}")
            return None

    def get_market_context_analysis(
        self, token_details: Optional[TokenDetails], original_prompt: str
    ) -> Optional[str]:
        """Analyze external market factors and competitive landscape"""
        try:
            # Base prompt with original user request
            base_prompt = f"""Original Request: "{original_prompt}"

"""
            # Add token data if available
            if token_details:
                base_prompt += f"""Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Category: Based on description: "{token_details['description']}"
"""

            prompt = (
                base_prompt
                + """As the Chief Market Strategist at a leading digital asset investment firm, provide strategic market intelligence for our institutional clients.

This analysis will be shared with:
- Hedge fund managers
- Private wealth clients
- Investment advisors
- Professional traders

Please provide your strategic market assessment covering:

1. Market Narrative Analysis:
   - Current state of this token's category/niche
   - Similar projects/tokens trending now
   - Drivers of interest in this type of project
   - Timing alignment with broader market trends

2. Chain Ecosystem Analysis:
   - Current state of the relevant ecosystem
   - Recent developments or challenges
   - Competitive positioning among chains
   - Technical advantages/disadvantages

3. Competitive Landscape:
   - Main competitors in this space
   - Market share distribution
   - Key differentiators
   - Dominant players and emerging threats"""
            )

            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": "You are the Chief Market Strategist at a prestigious digital asset investment firm. Your insights guide institutional investment strategies. Focus on macro trends, market dynamics, and strategic positioning.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating market context analysis: {e}")
            return None

    def generate_investment_report(
        self,
        token_details: Optional[TokenDetails],
        tokenomics_analysis: str,
        project_research: str,
        market_context: str,
        original_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a comprehensive investment report combining all analyses."""
        if not token_details:
            return "Could not generate report: Token details not available."

        try:
            prompt = f"""Create a comprehensive investment report for {token_details['name']} ({token_details['symbol']}) based on the following analyses:

TOKEN DETAILS:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap / FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Twitter Followers: {token_details['twitter_followers']:,}

TOKENOMICS ANALYSIS:
{tokenomics_analysis}

PROJECT RESEARCH:
{project_research}

MARKET CONTEXT:
{market_context}

Original user query: "{original_prompt}"

Format your response as a complete investment report with these sections:
1. Executive Summary (with clear investment recommendation)
2. Project Overview
3. Tokenomics Assessment
4. Market Position Analysis
5. Risk Factors
6. Investment Outlook (short, medium, and long term)
7. Key Metrics to Monitor

Be balanced, data-driven, and provide specific insights rather than generic statements."""

            default_system_prompt = "You are a cryptocurrency investment analyst creating comprehensive reports for investors. Your analysis is balanced, data-driven, and provides specific insights rather than generic statements. You always include both bull and bear perspectives. Output the investment report directly without repeating these instructions or explaining what you're going to do."
            
            # Use the LiteLLM utility instead of direct OpenAI call
            from app.utils.llm import generate_completion
            
            return generate_completion(
                prompt=prompt,
                system_prompt=system_prompt if system_prompt else default_system_prompt,
                model=DEFAULT_MODEL,
                temperature=0.7
            )

        except Exception as e:
            print(f"Error generating investment report: {e}")
            return f"Error generating investment report: {str(e)}"

    def get_coingecko_id_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract CoinGecko ID from natural language prompt"""
        try:
            perplexity_prompt = f"""Given this question about a cryptocurrency: "{prompt}"

Please identify:
1. Which cryptocurrency is being asked about
2. What is its exact CoinGecko ID (the ID used in CoinGecko's API)

Important notes about CoinGecko IDs:
- They are always lowercase
- They never contain special characters (only letters, numbers, and hyphens)
- Common examples: 'bitcoin', 'ethereum', 'olas', 'solana'
- For newer tokens, check their official documentation or CoinGecko listing

Format your response exactly like this example:
Cryptocurrency: Bitcoin
CoinGecko ID: bitcoin

Only provide these two lines, nothing else."""

            response = self.perplexity_client.chat.completions.create(
                model="sonar-reasoning",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency expert. Your task is to identify the specific cryptocurrency being discussed and provide its exact CoinGecko ID. Be precise and only return the requested format.",
                    },
                    {"role": "user", "content": perplexity_prompt},
                ],
            )

            response_text = response.choices[0].message.content
            for line in response_text.split("\n"):
                if line.startswith("CoinGecko ID:"):
                    raw_id = line.replace("CoinGecko ID:", "").strip()
                    clean_id = "".join(
                        c for c in raw_id.lower() if c.isalnum() or c == "-"
                    )
                    return clean_id

            return None

        except Exception as e:
            print(f"Error identifying CoinGecko ID: {e}")
            return None

    def run(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Run fundamental analysis and return structured response"""
        try:
            # Try to get token ID and details from CoinGecko
            token_id = self.get_coingecko_id_from_prompt(prompt)
            token_details = None
            if token_id:
                try:
                    token_details = self.get_token_details(token_id)
                except Exception as e:
                    token_details = None

            # Generate analyses (with or without token details)
            try:
                tokenomics_analysis = self.get_investment_analysis(token_details, prompt)
            except Exception:
                tokenomics_analysis = "Tokenomics analysis unavailable due to incomplete market data."

            try:
                project_research = self.get_project_research(token_details, prompt)
            except Exception:
                project_research = "Project research unavailable due to incomplete data."

            try:
                market_context = self.get_market_context_analysis(token_details, prompt)
            except Exception:
                market_context = "Market context analysis unavailable due to incomplete data."

            # Generate final report even if some analyses failed
            analysis = self.generate_investment_report(
                token_details,
                tokenomics_analysis,
                project_research,
                market_context,
                prompt,
                system_prompt,
            )

            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "analyses": {
                    "tokenomics": tokenomics_analysis,
                    "project": project_research,
                    "market": market_context,
                },
            }

            if token_details:
                metadata.update({
                    "token_details": token_details,
                    "token_id": token_id,
                })
            else:
                metadata["note"] = "Analysis generated with partial or no CoinGecko data"

            return {"response": analysis, "metadata": metadata}

        except Exception as e:
            return {"error": str(e)}


# added the following to have uniformity in the way we call tools
def run(prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
    return FundamentalAnalysis().run(prompt, system_prompt)
