# Eolas AI Chat Web App

A modern, responsive web-based chat interface for interacting with OpenAI models with optional tool integrations.

## Features

- Clean, modern UI with responsive design
- Real-time typing indicators and status updates via WebSockets
- Markdown and code syntax highlighting support
- Optional integration with trading analysis tools
- Token usage tracking
- Model selection (GPT-4o, GPT-4o Mini, GPT-3.5 Turbo)
- Tools on/off toggle

## Requirements

- Python 3.9+
- OpenAI API key

## Setup

1. Clone the repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Copy the example environment file and fill in your OpenAI API key:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file with your OpenAI API key.

## Running the App

Start the Flask server:

```
python ai_chat_app.py
```

Then open your browser to `http://localhost:5000`.

## Using the Interface

1. **Send a message**: Type in the input box and press Enter or click the send button.
2. **Toggle tools**: Click the "Tools" button to enable/disable tool selection.
3. **Change model**: Select a different model from the dropdown.
4. **Clear chat**: Click the "Clear" button to reset the conversation.

## Features Implemented

- [x] Chat history with clear user/assistant separation
- [x] Colorful, styled text with Markdown support
- [x] Scrollable conversation history
- [x] Loading/typing indicators
- [x] Input box at bottom of screen
- [x] Status bar (model, tokens used, status)
- [x] Tool usage indicators

## License

MIT 