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

def create_default_token_details(name: str, symbol: str = "", chain: str = "unknown") -> TokenDetails:
    """Create default token details when CoinGecko data is not available."""
    return TokenDetails(
        name=name,
        symbol=symbol.upper() if symbol else name.upper(),
        chain=chain,
        contract_address="",
        description="",
        market_cap=0.0,
        market_cap_fdv_ratio=0.0,
        price_change_24h=0.0,
        price_change_14d=0.0,
        twitter_followers=0,
        links={}
    )

def get_token_details(token_id: Optional[str], token_name: str) -> TokenDetails:
    """Get detailed information about a token from CoinGecko, or return default details if not found."""
    if not token_id:  # Only check for token_id, since we know COINGECKO_API_KEY exists
        return create_default_token_details(token_name)

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
        print(f"Error fetching token details from CoinGecko: {e}")
        return create_default_token_details(token_name)

def get_investment_analysis(token_details: TokenDetails, original_prompt: str) -> Optional[str]:
    """Get focused tokenomics and market sentiment analysis using GPT."""
    try:
        # Adjust prompt based on available data
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

        prompt = f"""As a forward-thinking tokenomics expert at a leading crypto venture capital firm, analyze this token's growth potential:

Original Request: "{original_prompt}"

Token: {token_details['name']} ({token_details['symbol']})
Key Metrics:
{metrics_text}

Focus on:
1. Growth Potential:
   - Market positioning and expansion opportunities
   - Potential catalysts in token distribution
   - Supply dynamics that could drive value
   - Comparison to successful projects at similar stages

2. Market Momentum:
   - Growth indicators and market sentiment
   - Social and community growth potential
   - Market conditions favoring growth

Highlight the unique opportunities and growth catalysts while acknowledging any temporary challenges."""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a visionary tokenomics researcher at a prestigious crypto venture capital firm. You're known for identifying promising projects early and seeing potential where others see challenges. Your analyses focus on growth opportunities and catalysts that could drive success, while still maintaining analytical credibility."
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

def get_project_research(token_details: TokenDetails, original_prompt: str) -> Optional[str]:
    """Research the project using Perplexity API to analyze links and provide insights."""
    try:
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
        
        links_text = "\n".join([f"- {url}" for url in research_links]) if research_links else "No official links available"
        
        prompt = f"""As an innovation-focused blockchain researcher, analyze this project's potential:

Original Request: "{original_prompt}"

Project: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Contract: {token_details['contract_address']}
Description: {token_details['description']}

Available Sources:
{links_text}

Explore and highlight:
1. Innovation & Vision:
   - Unique technological advantages
   - Market problems being solved
   - Potential for industry disruption
   - Growth opportunities

2. Team & Development:
   - Development momentum
   - Community growth initiatives
   - Technical capabilities
   - Innovation pipeline

3. Future Catalysts:
   - Upcoming milestones
   - Partnership opportunities
   - Market expansion potential
   - Growth accelerators

Focus on the project's strengths and potential while acknowledging areas for improvement."""

        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are a forward-thinking blockchain researcher who specializes in identifying innovative projects with high potential. You focus on understanding how projects can succeed and grow, while maintaining analytical credibility. Your analysis emphasizes opportunities while acknowledging the path to achieving them."
            }, {
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating project research: {e}")
        return None

def get_market_context_analysis(token_details: TokenDetails, original_prompt: str) -> Optional[str]:
    """Analyze external market factors and competitive landscape."""
    try:
        prompt = f"""As a growth-focused Market Strategist, analyze this token's market potential:

Original Request: "{original_prompt}"

Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Category: Based on description: "{token_details['description']}"

Provide strategic insights on:

1. Market Opportunity:
   - Growth potential in token's category/niche
   - Emerging market trends supporting success
   - Adoption catalysts
   - Market expansion possibilities

2. Ecosystem Advantages:
   - Strategic partnerships and synergies
   - Developer ecosystem potential
   - Community growth opportunities
   - {token_details['chain']} ecosystem benefits

3. Development Momentum:
   - Recent achievements
   - Promising roadmap milestones
   - Innovation pipeline
   - Growth accelerators

4. Competitive Edge:
   - Unique value propositions
   - Market positioning strengths
   - First-mover advantages
   - Growth opportunities vs competitors

Focus on identifying paths to success while acknowledging the steps needed to achieve them."""

        response = perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are a growth-focused Market Strategist who excels at identifying market opportunities and expansion potential. Your analysis emphasizes paths to success while maintaining credibility through data-driven insights. You help others understand how projects can capture market opportunities."
            }, {
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating market context analysis: {e}")
        return None

def generate_investment_report(token_details: TokenDetails, tokenomics_analysis: str, project_research: str, market_context: str, original_prompt: str) -> str:
    """Generate a comprehensive investment report combining all analyses."""
    try:
        prompt = f"""As a visionary Chief Investment Officer, synthesize this research into a growth-focused investment thesis:

Original Request: "{original_prompt}"

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

Synthesize a growth-focused thesis covering:
- Key opportunities and growth catalysts
- Path to capturing market potential
- Strategic advantages and moats
- Milestones that could accelerate growth

Focus on how the project can succeed while acknowledging the steps needed to realize its potential."""

        completion = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a visionary Chief Investment Officer known for identifying high-potential opportunities early. Your investment theses focus on growth potential and paths to success, while maintaining credibility through data-driven analysis. You help others understand how projects can achieve their potential."
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
        
        # Try to get CoinGecko ID from prompt
        print("\nIdentifying token...")
        token_id = get_coingecko_id_from_prompt(prompt)
        
        # Extract token name from prompt if no CoinGecko ID found
        if not token_id:
            print("Could not find exact match on CoinGecko. Using token name from prompt.")
            # Use Perplexity to extract token name from prompt
            name_prompt = f"""Extract the name of the cryptocurrency or token from this prompt: "{prompt}"
            Respond with ONLY the token name, nothing else."""
            
            response = perplexity_client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{
                    "role": "system",
                    "content": "Extract only the token name from the prompt. Respond with just the name, no explanation."
                }, {
                    "role": "user",
                    "content": name_prompt
                }]
            )
            token_name = response.choices[0].message.content.strip()
        else:
            token_name = token_id.replace('-', ' ').title()
        
        # Get token details
        print(f"\nFetching details for {token_name}...")
        token_details = get_token_details(token_id, token_name)
        
        # Generate analyses
        print("\nGenerating tokenomics analysis...")
        tokenomics_analysis = get_investment_analysis(token_details, prompt)
        
        print("\nResearching project details...")
        project_research = get_project_research(token_details, prompt)
        
        print("\nAnalyzing market context...")
        market_context = get_market_context_analysis(token_details, prompt)
        
        if not all([tokenomics_analysis, project_research, market_context]):
            print("Error generating complete analysis. Please try again.")
            return
        
        # Generate final report
        print("\nGenerating comprehensive investment report...")
        final_report = generate_investment_report(token_details, tokenomics_analysis, project_research, market_context, prompt)
        
        # Print the final report
        print("\n" + "="*80)
        print("INVESTMENT REPORT")
        print("="*80 + "\n")
        print(final_report)
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{token_name.lower().replace(' ', '_')}_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Investment Report for {token_details['name']} ({token_details['symbol']})\n")
            f.write("="*80 + "\n\n")
            f.write("Original Request: {prompt}\n\n")
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