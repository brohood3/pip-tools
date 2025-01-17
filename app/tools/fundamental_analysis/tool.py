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
    links: Dict[str, Any]

class FundamentalAnalysis:
    def __init__(self):
        """Initialize OpenAI client and API keys"""
        self.openai_client = OpenAI()
        self.perplexity_client = OpenAI(
            api_key=os.getenv('PERPLEXITY_API_KEY'),
            base_url="https://api.perplexity.ai"
        )
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')

    def get_token_details(self, token_id: str) -> Optional[TokenDetails]:
        """Get detailed information about a token from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
            headers = {
                "accept": "application/json",
                "x-cg-demo-api-key": self.coingecko_api_key
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            platforms = data.get('platforms', {})
            chain = next(iter(platforms.keys())) if platforms else 'ethereum'
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
            print(f"Error fetching token details: {e}")
            return None

    def get_investment_analysis(self, token_details: Optional[TokenDetails], original_prompt: str) -> Optional[str]:
        """Get focused tokenomics and market sentiment analysis"""
        try:
            # Base prompt with original user request
            base_prompt = f"""Original Request: "{original_prompt}"

"""
            # Add market data if available
            if token_details:
                base_prompt += f"""Token: {token_details['name']} ({token_details['symbol']})
Key Metrics:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Social Following: {token_details['twitter_followers']:,} Twitter followers
"""
            
            prompt = base_prompt + """As a seasoned tokenomics expert at a top crypto venture capital firm, analyze this investment opportunity for our institutional investors.

Your analysis should be suitable for sophisticated investors who:
- Understand DeFi fundamentals
- Are looking for detailed technical analysis
- Need clear risk/reward assessments
- Require institutional-grade due diligence

Please provide your VC firm's analysis covering:

1. Tokenomics Analysis:
   - Token distribution and supply dynamics
   - Market valuation assessment
   - Supply/demand dynamics
   - Potential dilution risks

2. Market Momentum Analysis:
   - Recent price action and trends
   - Market sentiment indicators
   - Social metrics and community engagement
   - Relative valuation metrics"""

            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are the head of tokenomics research at a prestigious crypto venture capital firm. Your analyses influence multi-million dollar investment decisions. Be thorough, technical, and unbiased in your assessment."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating analysis: {e}")
            return None

    def get_project_research(self, token_details: Optional[TokenDetails], original_prompt: str) -> Optional[str]:
        """Research the project using Perplexity API"""
        try:
            # Base prompt with original user request
            base_prompt = f"""Original Request: "{original_prompt}"

"""
            # Add project data if available
            if token_details:
                research_links = []
                important_link_types = ['homepage', 'blockchain_site', 'whitepaper', 'announcement_url', 'twitter_screen_name', 'telegram_channel_identifier', 'github_url']
                
                for link_type, urls in token_details['links'].items():
                    if link_type in important_link_types:
                        if isinstance(urls, list):
                            research_links.extend([url for url in urls if url])
                        elif isinstance(urls, str) and urls:
                            if link_type == 'telegram_channel_identifier':
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

            prompt = base_prompt + """As the lead blockchain researcher at a top-tier crypto investment fund, conduct comprehensive due diligence for our portfolio managers.

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

            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{
                    "role": "system",
                    "content": "You are a senior blockchain researcher at a $500M crypto fund. Your research directly influences investment allocation decisions. Maintain professional skepticism and support claims with evidence."
                }, {
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating project research: {e}")
            return None

    def get_market_context_analysis(self, token_details: Optional[TokenDetails], original_prompt: str) -> Optional[str]:
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

            prompt = base_prompt + """As the Chief Market Strategist at a leading digital asset investment firm, provide strategic market intelligence for our institutional clients.

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

            response = self.perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{
                    "role": "system",
                    "content": "You are the Chief Market Strategist at a prestigious digital asset investment firm. Your insights guide institutional investment strategies. Focus on macro trends, market dynamics, and strategic positioning."
                }, {
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating market context analysis: {e}")
            return None

    def generate_investment_report(self, token_details: Optional[TokenDetails], tokenomics_analysis: str, project_research: str, market_context: str, original_prompt: str) -> str:
        """Generate comprehensive investment report combining all analyses"""
        try:
            # Base prompt with original user request
            base_prompt = f"""Original Request: "{original_prompt}"

"""
            # Add market data if available
            if token_details:
                base_prompt += f"""Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Key Metrics:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Social Following: {token_details['twitter_followers']:,} Twitter followers
"""

            prompt = base_prompt + f"""As the Chief Investment Officer of a leading crypto investment firm, analyze our research findings and provide your investment thesis.

RESEARCH FINDINGS:

1. Tokenomics Analysis:
{tokenomics_analysis}

2. Project Research:
{project_research}

3. Market Context:
{market_context}

Based on this research, provide your investment thesis with clear sections. Structure your response like this:

# Investment Stance

[Main investment thesis backed by specific data points from all three analyses]

# Opportunities and Risks

[Key opportunities and risks analysis]

# Entry/Exit Strategy

[Strategic actionable recommendations]

# Key Metrics

[Important metrics to watch that could change your thesis]

Use clear line breaks between sections and paragraphs. Keep each section focused and concise."""

            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are the Chief Investment Officer at a prestigious crypto investment firm. Format your analysis with clear sections separated by line breaks. Make clear, opinionated investment recommendations backed by data. Be decisive but support all major claims with evidence."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating investment report: {e}")
            return None

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
                model="llama-3.1-sonar-large-128k-online",
                messages=[{
                    "role": "system",
                    "content": "You are a cryptocurrency expert. Your task is to identify the specific cryptocurrency being discussed and provide its exact CoinGecko ID. Be precise and only return the requested format."
                }, {
                    "role": "user",
                    "content": perplexity_prompt
                }]
            )
            
            response_text = response.choices[0].message.content
            for line in response_text.split('\n'):
                if line.startswith('CoinGecko ID:'):
                    raw_id = line.replace('CoinGecko ID:', '').strip()
                    clean_id = ''.join(c for c in raw_id.lower() if c.isalnum() or c == '-')
                    return clean_id
            
            return None
            
        except Exception as e:
            print(f"Error identifying CoinGecko ID: {e}")
            return None

    def run(self, prompt: str) -> Dict[str, Any]:
        """Run fundamental analysis and return structured response"""
        try:
            # Try to get token ID and details from CoinGecko
            token_id = self.get_coingecko_id_from_prompt(prompt)
            token_details = None
            if token_id:
                token_details = self.get_token_details(token_id)
            
            # Generate analyses (with or without token details)
            tokenomics_analysis = self.get_investment_analysis(token_details, prompt)
            project_research = self.get_project_research(token_details, prompt)
            market_context = self.get_market_context_analysis(token_details, prompt)
            
            if not all([tokenomics_analysis, project_research, market_context]):
                return {"error": "Error generating complete analysis. Please try again."}
            
            # Generate final report
            analysis = self.generate_investment_report(token_details, tokenomics_analysis, project_research, market_context, prompt)
            
            # Store all context in metadata
            metadata = {
                "prompt": prompt,
                "timestamp": datetime.now().isoformat(),
                "analyses": {
                    "tokenomics": tokenomics_analysis,
                    "project": project_research,
                    "market": market_context
                }
            }

            # Add token details to metadata if available
            if token_details:
                metadata.update({
                    "token_details": token_details,
                    "token_id": token_id,
                    "metrics": {
                        "market_cap": token_details["market_cap"],
                        "market_cap_fdv_ratio": token_details["market_cap_fdv_ratio"],
                        "price_change_24h": token_details["price_change_24h"],
                        "price_change_14d": token_details["price_change_14d"],
                        "twitter_followers": token_details["twitter_followers"]
                    },
                    "chain_info": {
                        "chain": token_details["chain"],
                        "contract": token_details["contract_address"]
                    },
                    "links": token_details["links"]
                })
            else:
                metadata["note"] = "Analysis generated without CoinGecko data"
            
            return {
                "analysis": analysis,
                "metadata": metadata
            }
            
        except Exception as e:
            return {"error": str(e)} 