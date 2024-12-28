import functools

import openai
import requests
import os
from typing import List, Dict, Optional, TypedDict, Tuple, Any, Callable
from openai import OpenAI

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

class APIClients:
    def __init__(self, api_keys: Any):
        self.coingecko_api_key = api_keys["coingecko"]
        self.openai_api_key = api_keys["openai"]
        self.perplexity_api_key = api_keys["perplexity"]
        
        if not all([self.coingecko_api_key, self.openai_api_key, self.perplexity_api_key]):
            raise ValueError("Missing required API keys in environment variables")
            
        self.openai_client = OpenAI()
        self.perplexity_client = OpenAI(
            api_key=self.perplexity_api_key,
            base_url="https://api.perplexity.ai"
        )

# Data Fetching Functions
def get_token_details(token_id: str) -> Optional[TokenDetails]:
    """
    Get detailed information about a token from CoinGecko
    Returns TokenDetails with key metrics and information
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token_id}"
        headers = {
            "accept": "application/json",
            "x-cg-demo-api-key": os.getenv('COINGECKO_API_KEY')
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Get first platform as chain and its contract address
        platforms = data.get('platforms', {})
        chain = next(iter(platforms.keys())) if platforms else 'ethereum'
        contract_address = platforms.get(chain, '') if platforms else ''
        
        # Get all links
        links = data.get('links', {})
        
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
            links=links
        )
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching token details: {e}")
        return None

# Analysis Generation Functions
def get_investment_analysis(clients: APIClients, token_details: TokenDetails) -> Optional[str]:
    """
    Get focused tokenomics and market sentiment analysis using GPT
    Returns the raw analysis text
    """
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

        completion = clients.openai_client.chat.completions.create(
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

def get_project_research(clients: APIClients, token_details: TokenDetails) -> Optional[str]:
    """
    Research the project using Perplexity API to analyze links and provide insights
    Returns the raw research text
    """
    try:
        # Prepare relevant links for research
        research_links = []
        important_link_types = ['homepage', 'blockchain_site', 'whitepaper', 'announcement_url', 'twitter_screen_name', 'telegram_channel_identifier', 'github_url', 'youtube_url', 'discord_url', 'linkedin_url', 'facebook_url', 'instagram_url', 'reddit_url', 'telegram_url', 'tiktok_url', 'website', 'blog', 'telegram', 'discord', 'reddit', 'linkedin', 'facebook', 'instagram', 'tiktok', 'youtube']
        
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
        
        prompt = f"""As the lead blockchain researcher at a top-tier crypto investment fund, conduct comprehensive due diligence for our portfolio managers:

Project: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Contract: {token_details['contract_address']}
Description: {token_details['description']}

Available Sources:
{links_text}

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
   - Upcoming features or releases
"""

        response = clients.perplexity_client.chat.completions.create(
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

def get_market_context_analysis(clients: APIClients, token_details: TokenDetails) -> Optional[str]:
    """
    Analyze external market factors, narratives, and competitive landscape using Perplexity
    Returns the raw analysis text
    """
    try:
        prompt = f"""As the Chief Market Strategist at a leading digital asset investment firm, provide strategic market intelligence for our institutional clients:

Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Category: Based on description: "{token_details['description']}"

This analysis will be shared with:
- Hedge fund managers
- Private wealth clients
- Investment advisors
- Professional traders

Please provide your strategic market assessment covering:

1. Market Narrative Analysis:
   - What is the current state of this token's category/niche in the market?
   - Are similar projects/tokens trending right now?
   - What's driving interest in this type of project?
   - How does the timing align with broader market trends?

2. Chain Ecosystem Analysis:
   - What is the current state of {token_details['chain']} ecosystem?
   - Recent developments or challenges in the chain?
   - How does this chain compare to competitors for this type of project?
   - What are the advantages/disadvantages of launching on this chain?

3. Competitive Landscape:
   - Who are the main competitors in this space?
   - What's the market share distribution?
   - What are the key differentiators between projects?
   - Are there any dominant players or emerging threats?

Please use real-time market data and recent developments in your analysis."""

        response = clients.perplexity_client.chat.completions.create(
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

# Report Generation Function
def generate_investment_report(clients: APIClients, token_details: TokenDetails, tokenomics_analysis: str, project_research: str, market_context: str) -> Optional[str]:
    """
    Generate a concise investment report aggregating all analyses
    Returns the formatted report text
    """
    try:
        prompt = f"""As the Investment Committee Chair at a leading crypto investment firm, synthesize our research team's findings into an executive summary for the board:

Token: {token_details['name']} ({token_details['symbol']})
Chain: {token_details['chain']}
Market Cap: ${token_details['market_cap']:,.2f}

TOKENOMICS ANALYSIS:
{tokenomics_analysis}

PROJECT RESEARCH:
{project_research}

MARKET CONTEXT:
{market_context}

This report will be presented to:
- Board members
- Executive leadership
- Key stakeholders
- Investment partners

Based on our team's comprehensive analyses:

1. Investment Outlook (2-3 sentences)
   - Overall sentiment
   - Key market position
   - Growth potential

2. Key Catalysts (3-4 bullet points)
   - Upcoming developments
   - Market opportunities
   - Competitive advantages

3. Risk Factors (3-4 bullet points)
   - Market risks
   - Project-specific risks
   - External threats

4. Investment Recommendations
   - Entry strategy
   - Position sizing considerations
   - Key metrics to monitor
   - Time horizon"""

        completion = clients.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are the Investment Committee Chair at a prestigious crypto investment firm. Your recommendations directly influence multi-million dollar allocation decisions. Be decisive, clear, and thorough in your risk/reward assessment."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.7
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating investment report: {e}")
        return None

def get_coingecko_id_from_prompt(clients: APIClients, prompt: str) -> Optional[str]:
    """
    Use Perplexity to identify the correct CoinGecko ID from a natural language prompt
    Returns the CoinGecko ID if found, None otherwise
    """
    try:
        # Create a prompt that asks specifically for the CoinGecko ID
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

Only provide these two lines, nothing else. Do not add any citations, references, or extra characters."""

        response = clients.perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are a cryptocurrency expert. Your task is to identify the specific cryptocurrency being discussed and provide its exact CoinGecko ID. Be precise and only return the requested format. Never add citations or references."
            }, {
                "role": "user",
                "content": perplexity_prompt
            }]
        )
        
        # Parse the response to extract the CoinGecko ID
        response_text = response.choices[0].message.content
        for line in response_text.split('\n'):
            if line.startswith('CoinGecko ID:'):
                # Clean the ID: lowercase, remove special chars except hyphens
                raw_id = line.replace('CoinGecko ID:', '').strip()
                clean_id = ''.join(c for c in raw_id.lower() if c.isalnum() or c == '-')
                return clean_id
        
        return None
        
    except Exception as e:
        print(f"Error identifying CoinGecko ID: {e}")
        return None

def get_general_market_analysis(clients: APIClients) -> Optional[str]:
    """
    Generate a general cryptocurrency market analysis using Bitcoin as a baseline
    Returns the analysis text
    """
    try:
        # Get Bitcoin details as market baseline
        btc_details = get_token_details('bitcoin')
        if not btc_details:
            print("Could not fetch Bitcoin market data for baseline analysis")
            return None

        prompt = f"""As the Chief Market Strategist at a leading digital asset investment firm, provide a comprehensive analysis of the current cryptocurrency market landscape:

Market Baseline (Bitcoin):
- Market Cap: ${btc_details['market_cap']:,.2f}
- 24h Change: {btc_details['price_change_24h']:.2f}%
- 14d Change: {btc_details['price_change_14d']:.2f}%

Please provide a thorough market analysis covering:

1. Current Market Environment
   - Overall market sentiment
   - Key market trends
   - Major narratives driving the market
   - Institutional vs retail participation

2. Market Opportunities
   - Emerging sectors in crypto
   - Areas of innovation
   - Potential growth catalysts
   - Market inefficiencies

3. Risk Assessment
   - Macro risks
   - Regulatory landscape
   - Technical considerations
   - Market structure concerns

4. Investment Strategy
   - Asset allocation considerations
   - Risk management approaches
   - Entry/exit strategies
   - Portfolio construction advice

Focus on providing actionable insights for sophisticated investors who understand the crypto market dynamics."""

        response = clients.perplexity_client.chat.completions.create(
            model="llama-3.1-sonar-large-128k-online",
            messages=[{
                "role": "system",
                "content": "You are the Chief Market Strategist at a prestigious digital asset investment firm. Your analysis guides institutional investment strategies across the cryptocurrency market. Be thorough, data-driven, and focus on actionable insights."
            }, {
                "role": "user",
                "content": prompt
            }]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating general market analysis: {e}")
        return None

MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]

def with_key_rotation(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # this is expected to be a KeyChain object,
        # although it is not explicitly typed as such
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()

        def execute() -> MechResponse:
            """Retry the function with a new key."""
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                # try with a new key again
                if retries_left["openai"] <= 0 and retries_left["openrouter"] <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), "", None, None, api_keys

        mech_response = execute()
        return mech_response

    return wrapper


@with_key_rotation
def run(
    prompt: str,
    api_keys: Any,
    **kwargs: Any,
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    # default response
    response = "Invalid response"

    try:
        clients = APIClients(api_keys)

        print(f"\nAnalyzing prompt: {prompt}")
        # Get CoinGecko ID from the prompt
        coin_id = get_coingecko_id_from_prompt(clients, prompt)
        if not coin_id:
            print("\nCould not identify a specific cryptocurrency from your prompt.")
            print("Generating general cryptocurrency market analysis instead...")
            
            analysis = get_general_market_analysis(clients)
            if analysis:
                print("\n=== CRYPTOCURRENCY MARKET ANALYSIS ===")
                print(analysis)
            return
            
        print(f"\nIdentified cryptocurrency with CoinGecko ID: {coin_id}")
        print("Fetching details...")
        
        details = get_token_details(coin_id)
        if not details:
            print("\nCould not fetch token details.")
            print("Generating general cryptocurrency market analysis instead...")
            
            analysis = get_general_market_analysis(clients)
            if analysis:
                print("\n=== CRYPTOCURRENCY MARKET ANALYSIS ===")
                print(analysis)
            return
            
        # Display basic token information
        print(f"\n{details['name']} ({details['symbol']})")
        print(f"Chain: {details['chain']}")
        print(f"Contract Address: {details['contract_address']}")
        print(f"Market Cap: ${details['market_cap']:,.2f}")
        print(f"Market Cap/FDV Ratio: {details['market_cap_fdv_ratio']:.2f}")
        print(f"24h Price Change: {details['price_change_24h']:.2f}%")
        print(f"14d Price Change: {details['price_change_14d']:.2f}%")
        print(f"Twitter Followers: {details['twitter_followers']:,}")
        
        # Display links
        print("\nLinks:")
        for link_type, urls in details['links'].items():
            if urls and isinstance(urls, list):
                valid_urls = [url for url in urls if url]
                if valid_urls:
                    print(f"{link_type.replace('_', ' ').title()}:")
                    for url in valid_urls:
                        print(f"- {url}")
            elif urls and isinstance(urls, str):
                print(f"{link_type.replace('_', ' ').title()}: {urls}")
            elif isinstance(urls, dict) and urls:
                print(f"{link_type.replace('_', ' ').title()}:")
                for sub_type, sub_urls in urls.items():
                    if sub_urls:
                        print(f"  {sub_type.title()}:")
                        for url in sub_urls:
                            if url:
                                print(f"  - {url}")
        
        # Display description
        print("\nDescription:")
        print(details['description'][:500] + "..." if len(details['description']) > 500 else details['description'])

        # Generate analyses
        print("\nGenerating Tokenomics & Market Analysis...")
        tokenomics_analysis = get_investment_analysis(clients, details)
        if tokenomics_analysis:
            print("\nTokenomics & Market Analysis:")
            response = tokenomics_analysis
        
        print("\nResearching Project Details...")
        project_research = get_project_research(clients, details)
        if project_research:
            print("\nProject Research:")
            response = project_research
            
        print("\nAnalyzing Market Context & Competition...")
        market_context = get_market_context_analysis(clients, details)
        if market_context:
            print("\nMarket Context Analysis:")
            response = market_context
            
        if all([tokenomics_analysis, project_research, market_context]):
            print("\nGenerating Investment Report...")
            report = generate_investment_report(clients, details, tokenomics_analysis, project_research, market_context)
            if report:
                print("\n=== INVESTMENT REPORT ===")
                response = report

    except Exception as e:
        print(f"An error occurred: {e}")

    return (
        response,
        "",
        None,
        None,
    )
