# Crypto Fundamental Analysis API

A powerful API that provides comprehensive fundamental analysis for cryptocurrency tokens, including tokenomics, project research, and market context.

## Features

- Token identification from natural language prompts
- Detailed tokenomics analysis
- Project research and ecosystem analysis
- Market context and competitive landscape
- Comprehensive investment reports
- Full API documentation with Swagger UI

## Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
COINGECKO_API_KEY=your_coingecko_key
OPENAI_API_KEY=your_openai_key
PERPLEXITY_API_KEY=your_perplexity_key
```

5. Run the server:
```bash
python api.py
```

The API will be available at `http://localhost:8000`
Documentation is available at `http://localhost:8000/docs`

## API Usage

### Generate Analysis

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze Bitcoin fundamentals"}'
```

### Response Format

```json
{
  "token_id": "bitcoin",
  "token_name": "Bitcoin",
  "token_symbol": "BTC",
  "report": "Comprehensive investment analysis...",
  "tokenomics_analysis": "Detailed tokenomics breakdown...",
  "project_research": "Project research findings...",
  "market_context": "Market analysis..."
}
```

## Deployment

### Deploy to Render

1. Create a new account on [Render](https://render.com)
2. Connect your GitHub repository
3. Click "New Web Service"
4. Select your repository
5. Render will automatically detect the configuration from `render.yaml`
6. Add your environment variables in the Render dashboard:
   - COINGECKO_API_KEY
   - OPENAI_API_KEY
   - PERPLEXITY_API_KEY
7. Deploy!

Your API will be available at `https://<service-name>.onrender.com`

### Alternative Deployment Options

- **Heroku**: Use the provided `Procfile`
- **AWS Elastic Beanstalk**: Use the provided `Dockerrun.aws.json`
- **Google Cloud Run**: Use the provided `Dockerfile`

## Rate Limits & Usage

- Respects CoinGecko API rate limits
- OpenAI API usage based on your plan
- Perplexity API limits apply

## Security Notes

- Always use HTTPS in production
- Protect your API keys
- Consider adding API key authentication for your endpoints
- Monitor usage and costs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 