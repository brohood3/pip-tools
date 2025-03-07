# Eolas AI Chat Application (Vite Version)

A modern AI chat application with a React frontend (built with Vite) and Flask backend, powered by OpenAI.

## Project Structure

The project is divided into two main parts:

- **Frontend**: A React application built with Vite
- **Backend**: A Flask API server

## Features

- Real-time chat interface using WebSockets
- Support for multiple AI models
- Tool integration for enhanced capabilities
- Markdown rendering and code syntax highlighting
- Token usage tracking
- Responsive design

## Getting Started

### Prerequisites

- Node.js and npm for the frontend
- Python 3.8+ and pip for the backend
- OpenAI API key

### Running the Backend

1. Navigate to the backend directory:
   ```
   cd ../chat-app/backend
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure your OpenAI API key is set in the `.env` file.

4. Run the Flask server:
   ```
   PORT=8080 python ai_chat_app.py
   ```

### Running the Frontend

1. Install dependencies (if not already done):
   ```
   npm install
   ```

2. Start the development server:
   ```
   npm run dev
   ```

3. Open http://localhost:3000 in your browser.

## Development

### Frontend

The React frontend is organized into the following structure:

- `components/`: UI components
- `contexts/`: React context providers
- `services/`: API services
- `styles/`: Global styles

### Vite Advantages

- Faster development server with HMR
- Optimized production builds
- No need for complex configuration
- TypeScript support out of the box
- Better developer experience

## Building for Production

To build the frontend for production:

```
npm run build
```

This will create a `dist` directory with optimized production files.

## License

This project is licensed under the MIT License.
