import React from 'react';
import { useState, useEffect } from 'react';
import Header from './components/Header/Header.jsx';
import ChatContainer from './components/ChatContainer/ChatContainer.jsx';
import InputArea from './components/InputArea/InputArea.jsx';
import StatusBar from './components/StatusBar/StatusBar.jsx';
import { initializeSocket, disconnectSocket, sendMessage, getModels, setModel, resetUsage } from './services/api.jsx';
import './App.css';

function App() {
  const [messages, setMessages] = useState([
    { 
      role: 'system', 
      content: "Hello, I'm Eolas. How can I assist you today?\n\nI can help with:\n- Answering questions\n- Providing information\n- Analyzing data\n- And more!" 
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [status, setStatus] = useState('Connecting...');
  const [tokenCount, setTokenCount] = useState(0);
  const [useTools, setUseTools] = useState(true);
  const [currentModel, setCurrentModel] = useState('gpt-4o');
  const [isConnected, setIsConnected] = useState(false);

  // Initialize Socket.io connection
  useEffect(() => {
    // eslint-disable-next-line no-unused-vars
    const socketConnection = initializeSocket({
      onConnect: () => {
        setIsConnected(true);
        setStatus('Connected');
      },
      onDisconnect: () => {
        setIsConnected(false);
        setStatus('Disconnected');
      },
      onTypingIndicator: (data) => {
        setIsTyping(data.status === 'typing');
      },
      onStatusUpdate: (data) => {
        setStatus(data.status);
      }
    });

    // Fetch available models on load
    const fetchModels = async () => {
      try {
        const data = await getModels();
        if (data.current) {
          setCurrentModel(data.current);
        }
      } catch (error) {
        console.error('Error fetching models:', error);
      }
    };

    fetchModels();

    // Cleanup function
    return () => {
      disconnectSocket();
    };
  }, []);

  // Add a new message to the chat
  const addMessage = (role, content, metadata = {}) => {
    const newMessage = { role, content, ...metadata };
    setMessages(prevMessages => [...prevMessages, newMessage]);
  };

  // Handle sending a message
  const handleSendMessage = async (message) => {
    if (!message.trim() || isTyping) return;
    
    // Add user message to chat
    addMessage('user', message);
    
    // Set typing indicator and status
    setIsTyping(true);
    setStatus('Processing...');
    
    try {
      // Send message to backend API
      const response = await sendMessage(message, useTools);
      
      // Add assistant response to chat
      addMessage('assistant', response.text, {
        timestamp: response.timestamp,
        tool_used: response.tool_used
      });
      
      // Update token count if available
      if (response.usage) {
        setTokenCount(response.usage.total_tokens);
      }
      
      setStatus('Ready');
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage('system', `Error: ${error.message || 'Failed to communicate with the server'}`);
      setStatus('Error occurred');
    } finally {
      setIsTyping(false);
    }
  };

  // Handle clearing the chat
  const handleClearChat = () => {
    setMessages([
      { 
        role: 'system', 
        content: "Hello, I'm Eolas. How can I assist you today?\n\nI can help with:\n- Answering questions\n- Providing information\n- Analyzing data\n- And more!" 
      }
    ]);
    resetUsage().then(() => {
      setTokenCount(0);
    }).catch(error => {
      console.error('Error resetting usage:', error);
    });
  };

  // Handle toggling tools
  const handleToggleTools = () => {
    setUseTools(prev => !prev);
    setStatus(`Tools ${!useTools ? 'enabled' : 'disabled'}`);
  };

  // Handle model change
  const handleModelChange = async (model) => {
    try {
      await setModel(model);
      setCurrentModel(model);
      setStatus(`Model changed to ${model}`);
    } catch (error) {
      console.error('Error changing model:', error);
      setStatus('Error changing model');
    }
  };

  return (
    <div className="App">
      <Header 
        onClearChat={handleClearChat}
        onToggleTools={handleToggleTools}
        onModelChange={handleModelChange}
        useTools={useTools}
        currentModel={currentModel}
      />
      <main className="chat-app-container">
        <ChatContainer messages={messages} />
        <StatusBar 
          isTyping={isTyping}
          status={status}
          tokenCount={tokenCount}
        />
        <InputArea 
          onSendMessage={handleSendMessage} 
          disabled={isTyping || !isConnected} 
        />
      </main>
    </div>
  );
}

export default App;
