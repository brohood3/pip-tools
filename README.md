# Pip - AI Chat UI with Modular Tooling

Pip is a playful, modular AI chat interface that combines a modern Next.js frontend with a powerful Python backend. The system features a growing collection of specialized tools ("toys") that can be accessed both through the chat interface and as standalone API endpoints.

![Pip Chat UI](https://via.placeholder.com/800x450.png?text=Pip+AI+Chat+Interface)

## 🚀 Features

- **Playful Chat Interface**: Interact with Pip, a baby AI agent learning about the world
- **Modular Tool System**: Extensible architecture for adding new capabilities
- **Web3 Integration**: Connect your wallet for blockchain interactions
- **API Access**: Use tools as standalone endpoints for integration with other applications
- **Markdown Support**: Rich text formatting with code syntax highlighting
- **Media Support**: Display charts, videos, and other media in chat responses
- **Transaction Execution**: Execute blockchain transactions directly from the chat

## 🧩 Available Tools

Pip has access to various tools that can be used through the chat interface or as API endpoints:

- **Technical Analysis**: Generate charts and analysis for cryptocurrency pairs
- **Video Generator**: Create videos based on text prompts
- **Brian Transaction**: Execute blockchain transactions
- **Query Extract**: Extract specific information from data sources
- **Fundamental Analysis**: Analyze cryptocurrency fundamentals
- **Macro Outlook Analyzer**: Provide macroeconomic insights
- **Tool Selector**: Automatically select the appropriate tool for a given task

## 🛠️ Setup Instructions

### Prerequisites

- Node.js (v16+)
- Python (v3.9+)
- OpenAI API key
- (Optional) Replicate API key for video generation
- (Optional) Web3 provider for blockchain interactions

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pip-chat.git
   cd pip-chat
   ```

2. Set up Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   REPLICATE_API_TOKEN=your_replicate_api_key
   SECRET_KEY=your_secret_key_for_flask
   ```

4. Start the Python backend:
   ```bash
   python ai_chat_app.py
   ```
   The backend will run on http://localhost:8080

### Frontend Setup

1. Navigate to the chat-ui directory:
   ```bash
   cd chat-ui
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```
   The frontend will run on http://localhost:3000

## 🔌 API Endpoints

All tools are available as API endpoints, making it easy to integrate them into other applications.

### Main Chat Endpoint

```
POST /api/chat
```

Request body:
```json
{
  "message": "Analyze the price of Bitcoin",
  "model": "gpt-4o",
  "use_tools": true,
  "wallet_address": "0x..."  // Optional
}
```

### Process Chat with Tools

```
POST /api/process_chat
```

Request body:
```json
{
  "message": "Generate a technical analysis chart for ETH/USD",
  "model": "gpt-4o",
  "use_tools": true,
  "wallet_address": "0x..."  // Optional
}
```

Response:
```json
{
  "text": "I've created a technical analysis chart for ETH/USD...",
  "model": "gpt-4o",
  "timestamp": "2023-06-15T12:34:56.789Z",
  "tool_used": "technical_analysis",
  "reasoning": "...",
  "confidence": "high",
  "usage": { ... },
  "chart": "base64_encoded_image"  // Tool-specific data
}
```

### Tool-Specific Endpoints

Each tool can be accessed directly through its own endpoint pattern:

```
POST /api/tools/{tool_name}
```

For example:
```
POST /api/tools/technical_analysis
```

Request body:
```json
{
  "prompt": "Create a chart for BTC/USD with RSI and MACD",
  "model": "gpt-4o"  // Optional
}
```

## 🧠 Extending with New Tools

The system is designed to be easily extended with new tools:

1. Create a new tool module in the `app/tools/` directory
2. Implement the required `run()` function
3. Register the tool in `ai_chat_app.py`
4. The tool will automatically be available in both the chat interface and as an API endpoint

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for the GPT models
- Replicate for video generation capabilities
- The Web3 community for blockchain integration tools 