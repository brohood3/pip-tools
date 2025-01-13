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
        prompt = f"""As an objective tokenomics expert at a top crypto venture capital firm, provide a data-driven analysis of this token:

Token: {token_details['name']} ({token_details['symbol']})
Key Metrics:
- Market Cap: ${token_details['market_cap']:,.2f}
- Market Cap/FDV Ratio: {token_details['market_cap_fdv_ratio']:.2f}
- 24h Price Change: {token_details['price_change_24h']:.2f}%
- 14d Price Change: {token_details['price_change_14d']:.2f}%
- Social Following: {token_details['twitter_followers']:,} Twitter followers

Focus on:
1. Tokenomics Analysis:
   - Evaluate the Market Cap/FDV ratio ({token_details['market_cap_fdv_ratio']:.2f}) against industry standards
   - Token distribution patterns and implications
   - Supply dynamics and potential impacts
   - Compare metrics to both successful and failed projects

2. Market Performance:
   - Analyze 24h vs 14d price action ({token_details['price_change_24h']:.2f}% vs {token_details['price_change_14d']:.2f}%)
   - Evaluate social metrics quality ({token_details['twitter_followers']:,} followers)
   - Market conditions impact assessment

Let the data guide your analysis. Highlight both strengths and weaknesses with specific evidence."""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a data-driven tokenomics researcher at a prestigious crypto venture capital firm. Your analyses are known for being thorough and objective, letting the metrics guide your conclusions. You've seen both successes and failures, and you understand that each project needs to be evaluated on its own merits. Your reputation comes from making accurate calls regardless of market sentiment."
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
        
        prompt = f"""As a seasoned blockchain investigator, conduct a thorough due diligence that challenges this project's claims:

Project: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Contract: {token_details['contract_address']}
Description: {token_details['description']}

Available Sources:
{links_text}

Investigate and expose:
1. Project Viability:
   - What crucial problems are they ignoring?
   - How realistic are their claims?
   - Which competitors are better positioned?

2. Team & Technology:
   - Red flags in development activity
   - Concerning patterns in community engagement
   - Technical limitations they're not addressing

3. Risk Assessment:
   - Potential points of failure
   - Regulatory vulnerabilities
   - Competition threats

Don't accept marketing claims at face value. Look for inconsistencies and hidden risks."""

        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are a battle-hardened blockchain investigator who has exposed numerous crypto scams and failures. Your research is known for challenging assumptions and finding hidden risks. You take pride in protecting investors through rigorous skepticism and detailed investigation."
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
        prompt = f"""As the Chief Investment Officer of a leading crypto investment firm, synthesize our research into an actionable investment thesis:

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

Based on this comprehensive research, provide:
- Clear investment thesis supported by specific data points
- Balanced risk/reward assessment
- Key metrics or events to monitor
- Specific conditions that would strengthen or invalidate the thesis

Your stance should reflect the weight of evidence, whether positive, negative, or mixed."""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a pragmatic Chief Investment Officer who bases decisions purely on evidence. Your investment theses are known for their accuracy because you let data drive the narrative, not market sentiment or personal bias. You're equally comfortable being bullish, bearish, or neutral when the evidence supports that position."
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