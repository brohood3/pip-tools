"""
A script for comprehensive fundamental analysis of cryptocurrency tokens.
Uses CoinGecko, OpenAI, and Perplexity APIs for data and analysis.
"""

import os
from typing import Dict, List, Optional, TypedDict
from datetime import datetime
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')

if not all([COINGECKO_API_KEY, OPENAI_API_KEY, PERPLEXITY_API_KEY]):
    raise ValueError("Missing required API keys in environment variables")

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
perplexity_client = OpenAI(
    api_key=PERPLEXITY_API_KEY,
    base_url="https://api.perplexity.ai"
)

# Type Definitions
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

def get_token_details(token_id: str) -> Optional[TokenDetails]:
    """Get detailed information about a token from CoinGecko."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": COINGECKO_API_KEY
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Get first platform as chain and its contract address
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

def get_investment_analysis(token_details: TokenDetails) -> Optional[str]:
    """Get focused tokenomics and market sentiment analysis using GPT."""
    try:
        prompt = f"""As a seasoned tokenomics expert at a top crypto venture capital firm, analyze this token for our institutional investors:

Token: {token_details['name']} ({token_details['symbol']})
Key Metrics:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Social Following: {token_details['twitter_followers']:,} Twitter followers

Your analysis should be suitable for sophisticated investors who:
- Understand DeFi fundamentals
- Are looking for detailed technical analysis
- Need clear risk/reward assessments
- Require institutional-grade due diligence

Please provide your VC firm's analysis in the following format:

1. Tokenomics Deep Dive:
   - Analyze the Market Cap/FDV ratio of {token_details['market_cap_fdv_ratio']:.2f}
   - What does this ratio suggest about token distribution and future dilution?
   - Compare to industry standards and identify potential red flags
   - Estimate locked/circulating supply implications

2. Market Momentum Analysis:
   - Interpret the 24h ({token_details['price_change_24h']:.2f}%) vs 14d ({token_details['price_change_14d']:.2f}%) price action
   - What does this trend suggest about market sentiment?
   - Analyze social metrics impact (Twitter following of {token_details['twitter_followers']:,})
   - Compare market cap to social engagement ratio"""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
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

def get_project_research(token_details: TokenDetails) -> Optional[str]:
    """Research the project using Perplexity API to analyze links and provide insights."""
    try:
        # Prepare relevant links for research
        research_links = []
        important_link_types = ['homepage', 'blockchain_site', 'whitepaper', 'announcement_url', 'twitter_screen_name']
        
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
        
        prompt = f"""As the lead blockchain researcher at a top-tier crypto investment fund, conduct comprehensive due diligence:

Project: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Contract: {token_details['contract_address']}
Description: {token_details['description']}

Available Sources:
{links_text}

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

        response = perplexity_client.chat.completions.create(
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

def get_market_context_analysis(token_details: TokenDetails) -> Optional[str]:
    """Analyze external market factors and competitive landscape."""
    try:
        prompt = f"""As the Chief Market Strategist at a leading digital asset investment firm, provide strategic market intelligence:

Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Category: Based on description: "{token_details['description']}"

Please provide your strategic market assessment covering:

1. Market Narrative Analysis:
   - Current state of this token's category/niche
   - Similar projects/tokens trending
   - Drivers of interest
   - Timing with broader market trends

2. Chain Ecosystem Analysis:
   - Current state of {token_details['chain']} ecosystem
   - Recent developments or challenges
   - Competitive advantages/disadvantages

3. Competitive Landscape:
   - Main competitors
   - Market share distribution
   - Key differentiators
   - Dominant players or emerging threats"""

        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are the Chief Market Strategist at a prestigious digital asset investment firm. Focus on macro trends, market dynamics, and strategic positioning."
            }, {
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating market context analysis: {e}")
        return None

def generate_investment_report(token_details: TokenDetails, tokenomics_analysis: str, project_research: str, market_context: str) -> str:
    """Generate a comprehensive investment report combining all analyses."""
    try:
        prompt = f"""As the Chief Investment Officer of a leading crypto investment firm, analyze our research findings and provide your investment thesis:

Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Key Metrics:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Social Following: {token_details['twitter_followers']:,} Twitter followers

RESEARCH FINDINGS:

1. Tokenomics Analysis:
{tokenomics_analysis}

2. Project Research:
{project_research}

3. Market Context:
{market_context}

Based on this research, provide your investment thesis and recommendations. Focus on:
- Clear investment stance backed by specific data points
- Most compelling opportunities and critical risks
- Actionable entry/exit strategies
- Key metrics that would change your thesis"""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are the Chief Investment Officer at a prestigious crypto investment firm. Make clear, opinionated investment recommendations backed by data. Be decisive but support all major claims with evidence."
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

def get_coingecko_id_from_prompt(prompt: str) -> Optional[str]:
    """Use Perplexity to identify the correct CoinGecko ID from a natural language prompt."""
    try:
        # Create a prompt that asks specifically for the CoinGecko ID
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

Input: "What about some random token?"
Output:
Cannot determine ID

Format your response exactly like the examples above. Do not add any citations, references, or footnotes."""

        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are a CoinGecko API expert. Your task is to identify the exact CoinGecko API ID for cryptocurrencies. Be precise and conservative - if unsure about the exact ID, respond with 'Cannot determine ID'. Never guess or make up IDs. Do not add citations or references to your response."
            }, {
                "role": "user",
                "content": perplexity_prompt
            }]
        )
        
        # Print the raw response for debugging
        response_text = response.choices[0].message.content
        print("\nPerplexity Response:")
        print("-" * 40)
        print(response_text)
        print("-" * 40)
        
        # Check for explicit uncertainty
        if "Cannot determine ID" in response_text:
            print("Token could not be identified with certainty")
            return None
        
        coingecko_id = None
        for line in response_text.split('\n'):
            if line.startswith('CoinGecko ID:'):
                # Remove the "CoinGecko ID:" prefix and clean the ID
                raw_id = line.replace('CoinGecko ID:', '').strip()
                # Remove any citation markers like [1], [2], etc.
                raw_id = raw_id.split('[')[0].strip('.')
                # Clean the ID to only allow alphanumeric and hyphens
                coingecko_id = ''.join(c for c in raw_id.lower() if c.isalnum() or c == '-')
                print(f"\nExtracted ID: {coingecko_id}")
                break
        
        if not coingecko_id:
            print("No CoinGecko ID found in response")
            return None
            
        # Verify the ID exists by making a test request
        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": COINGECKO_API_KEY
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Warning: Could not verify token ID '{coingecko_id}' exists in CoinGecko")
            return None
            
        return coingecko_id
        
    except Exception as e:
        print(f"Error identifying CoinGecko ID: {e}")
        return None

def main():
    """Main function to run the analysis."""
    try:
        # Get prompt from command line arguments or user input
        import sys
        if len(sys.argv) > 1:
            prompt = ' '.join(sys.argv[1:])
        else:
            prompt = input("Enter your analysis prompt (e.g., 'Analyze Bitcoin fundamentals'): ")
        
        print(f"\nAnalyzing prompt: {prompt}")
        
        # Get CoinGecko ID from prompt
        print("\nIdentifying token...")
        token_id = get_coingecko_id_from_prompt(prompt)
        if not token_id:
            print("Could not determine which token to analyze. Please specify a valid token in your prompt.")
            return
        
        # Get token details
        print(f"\nFetching details for {token_id}...")
        token_details = get_token_details(token_id)
        if not token_details:
            print(f"Could not fetch details for token {token_id}. Please verify the token exists.")
            return
        
        # Generate analyses
        print("\nGenerating tokenomics analysis...")
        tokenomics_analysis = get_investment_analysis(token_details)
        
        print("\nResearching project details...")
        project_research = get_project_research(token_details)
        
        print("\nAnalyzing market context...")
        market_context = get_market_context_analysis(token_details)
        
        if not all([tokenomics_analysis, project_research, market_context]):
            print("Error generating complete analysis. Please try again.")
            return
        
        # Generate final report
        print("\nGenerating comprehensive investment report...")
        final_report = generate_investment_report(token_details, tokenomics_analysis, project_research, market_context)
        
        # Print the final report
        print("\n" + "="*80)
        print("INVESTMENT REPORT")
        print("="*80 + "\n")
        print(final_report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{token_id}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Investment Report for {token_details['name']} ({token_details['symbol']})\n")
            f.write("="*80 + "\n\n")
            f.write("TOKENOMICS ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(tokenomics_analysis + "\n\n")
            f.write("PROJECT RESEARCH\n")
            f.write("-"*80 + "\n")
            f.write(project_research + "\n\n")
            f.write("MARKET CONTEXT\n")
            f.write("-"*80 + "\n")
            f.write(market_context + "\n\n")
            f.write("FINAL INVESTMENT THESIS\n")
            f.write("-"*80 + "\n")
            f.write(final_report)
        
        print(f"\nFull report saved to {filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 