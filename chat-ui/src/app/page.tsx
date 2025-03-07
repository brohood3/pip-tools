'use client'

import { ConnectButton } from "@/components/ConnectButton";
import { ActionButtonList } from "@/components/ActionButtonList";
import { useState, useRef, useEffect } from 'react';
import SimpleChatInput from '@/components/SimpleChatInput';
import { useAccount } from 'wagmi';

// Define the message type
interface Message {
  role: string;
  content: string;
  toolUsed?: string | null;
  timestamp?: string;
  confidence?: string;
  reasoning?: string;
}

export default function Home() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: "Hello, I'm Eolas. How can I assist you today?" }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [model, setModel] = useState("gpt-4o");
  const [useTools, setUseTools] = useState(true);
  const [tokenUsage, setTokenUsage] = useState({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Get wallet connection state
  const { address, isConnected } = useAccount();

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = async (message: string) => {
    // Add user message to chat
    setMessages(prev => [...prev, { role: 'user', content: message }]);
    setIsTyping(true);
    
    try {
      // Prepare message history - exclude the initial greeting and last user message
      // which we'll send separately
      const messageHistory = messages.slice(1).map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      if (messageHistory.length > 10) {
        // Limit history to last 10 messages to avoid token limits
        messageHistory.splice(0, messageHistory.length - 10);
      }
      
      // Call our API endpoint
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message, 
          model,
          use_tools: useTools,
          wallet_address: address,
          message_history: messageHistory
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      // Update token usage if available
      if (data.usage) {
        setTokenUsage(data.usage);
      }

      // Add assistant response
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: data.text || "Sorry, I couldn't process that request.",
        toolUsed: data.tool_used || null,
        timestamp: data.timestamp,
        confidence: data.confidence,
        reasoning: data.reasoning
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: "Sorry, there was an error processing your request." 
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  // Clear chat history
  const handleClearChat = () => {
    setMessages([
      { role: 'assistant', content: "Hello, I'm Eolas. How can I assist you today?" }
    ]);
  };

  return (
    <div className="pages">
      <div style={{ fontSize: '4rem', marginBottom: '1rem', color: '#6e48aa' }}>
        <i className="fas fa-robot"></i>
      </div>
      <h1>AI Chat UI</h1>

      {/* Header controls */}
      <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: '10px', margin: '20px 0' }}>
        {/* Wallet connection */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <ConnectButton />
          {isConnected && (
            <div style={{ 
              backgroundColor: '#f0f0f0', 
              padding: '10px', 
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              gap: '5px'
            }}>
              <i className="fas fa-wallet"></i>
              <span style={{ fontFamily: 'monospace' }}>
                {`${address?.slice(0, 6)}...${address?.slice(-4)}`}
              </span>
            </div>
          )}
        </div>
        
        {/* Model selector */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '5px',
          backgroundColor: 'white',
          border: '1px solid #ddd',
          borderRadius: '8px',
          padding: '0 8px'
        }}>
          <i className="fas fa-microchip"></i>
          <select 
            value={model}
            onChange={(e) => setModel(e.target.value)}
            style={{
              border: 'none',
              background: 'transparent',
              padding: '10px 0',
              fontSize: '14px'
            }}
          >
            <option value="gpt-4o">GPT-4o</option>
            <option value="gpt-4o-mini">GPT-4o Mini</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          </select>
        </div>
        
        {/* Tools toggle */}
        <button 
          onClick={() => setUseTools(!useTools)}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            padding: '10px',
            backgroundColor: useTools ? '#6e48aa' : '#f0f0f0',
            color: useTools ? 'white' : 'black',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          <i className="fas fa-wrench"></i>
          <span>Tools: {useTools ? 'On' : 'Off'}</span>
        </button>
        
        {/* Clear chat button */}
        <button 
          onClick={handleClearChat}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            padding: '10px',
            backgroundColor: '#f0f0f0',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer'
          }}
        >
          <i className="fas fa-trash"></i>
          <span>Clear</span>
        </button>
        
        {/* Action buttons for wallet */}
        <ActionButtonList />
      </div>
      
      {/* Chat UI */}
      <div className="chat-container" style={{ marginTop: '10px', border: '1px solid #ddd', borderRadius: '8px', padding: '10px', maxWidth: '800px', margin: '0 auto' }}>
        <div className="message-container" style={{ height: '400px', overflowY: 'auto', marginBottom: '10px' }}>
          {messages.map((message, index) => (
            <div 
              key={index} 
              className={`message ${message.role}`}
              style={{ 
                display: 'flex', 
                margin: '10px 0',
                flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                alignItems: 'flex-start'
              }}
            >
              <div 
                className={`avatar ${message.role}-avatar`}
                style={{ 
                  width: '36px', 
                  height: '36px', 
                  borderRadius: '50%', 
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: message.role === 'user' ? '#007bff' : '#6e48aa',
                  color: 'white',
                  marginRight: message.role === 'user' ? '0' : '10px',
                  marginLeft: message.role === 'user' ? '10px' : '0'
                }}
              >
                {message.role === 'user' ? 'U' : 'AI'}
              </div>
              <div 
                className="message-content"
                style={{ 
                  padding: '10px',
                  borderRadius: '8px',
                  backgroundColor: message.role === 'user' ? '#007bff' : 'white',
                  color: message.role === 'user' ? 'white' : 'black',
                  border: message.role === 'assistant' ? '1px solid #ddd' : 'none',
                  maxWidth: '70%'
                }}
              >
                <div>
                  {message.content.split('\n').map((text, i) => (
                    <p key={i} style={{ margin: '0 0 8px 0' }}>{text}</p>
                  ))}
                </div>
                
                {message.toolUsed && (
                  <div style={{ 
                    fontSize: '12px', 
                    marginTop: '5px', 
                    backgroundColor: '#f8f0ff',
                    padding: '8px',
                    borderRadius: '6px',
                    border: '1px solid #e0d0ff'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '3px' }}>
                      <i className="fas fa-tools"></i>
                      <span style={{ fontWeight: 'bold' }}>Tool: {message.toolUsed}</span>
                      {message.confidence && (
                        <span style={{ 
                          marginLeft: 'auto', 
                          backgroundColor: 
                            message.confidence === 'high' ? '#c8e6c9' : 
                            message.confidence === 'medium' ? '#fff9c4' : '#ffcdd2',
                          padding: '2px 6px',
                          borderRadius: '3px',
                          fontSize: '10px'
                        }}>
                          {message.confidence.toUpperCase()}
                        </span>
                      )}
                    </div>
                    {message.reasoning && (
                      <div style={{ fontSize: '11px', marginTop: '3px' }}>
                        <span style={{ fontStyle: 'italic' }}>Reasoning: {message.reasoning}</span>
                      </div>
                    )}
                  </div>
                )}
                
                {message.timestamp && (
                  <div style={{ fontSize: '11px', marginTop: '4px', opacity: 0.7, textAlign: 'right' }}>
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="typing-indicator" style={{ display: 'flex', padding: '10px', gap: '4px' }}>
              <div style={{ height: '8px', width: '8px', backgroundColor: '#bbb', borderRadius: '50%', animation: 'blink 1.4s infinite both' }}></div>
              <div style={{ height: '8px', width: '8px', backgroundColor: '#bbb', borderRadius: '50%', animation: 'blink 1.4s infinite both 0.2s' }}></div>
              <div style={{ height: '8px', width: '8px', backgroundColor: '#bbb', borderRadius: '50%', animation: 'blink 1.4s infinite both 0.4s' }}></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <SimpleChatInput onSendMessage={handleSendMessage} />
        
        {/* Token usage counter */}
        <div style={{ fontSize: '12px', textAlign: 'right', color: '#888', marginTop: '5px' }}>
          Tokens: {tokenUsage.total_tokens} (Prompt: {tokenUsage.prompt_tokens}, Completion: {tokenUsage.completion_tokens})
        </div>
      </div>

      <style jsx global>{`
        @keyframes blink {
          0% { opacity: 0.1; }
          20% { opacity: 1; }
          100% { opacity: 0.1; }
        }
      `}</style>
    </div>
  );
}