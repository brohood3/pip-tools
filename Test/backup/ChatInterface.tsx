'use client';

import { useEffect, useState, useRef } from 'react';
import { useAccount } from 'wagmi';
import { marked } from 'marked';
import hljs from 'highlight.js';
import { ConnectButton } from './ConnectButton';
import { useSocket } from '@/utils/socket';

// Fix TypeScript errors for marked options
declare module 'marked' {
  interface MarkedOptions {
    highlight?: (code: string, lang: string) => string;
  }
}

interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: {
    timestamp?: string;
    tool_used?: string;
    usage?: {
      total_tokens: number;
    };
  };
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'system',
      content: "Hello, I'm Eolas. Please connect your wallet to start chatting.",
    },
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [useTools, setUseTools] = useState(true);
  const [tokenCount, setTokenCount] = useState(0);
  const [status, setStatus] = useState('Ready');
  const [isTyping, setIsTyping] = useState(false);
  const [currentModel, setCurrentModel] = useState('gpt-4o');
  const [isConnected, setIsSocketConnected] = useState(false);
  const messageContainerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { address, isConnected: isWalletConnected } = useAccount();
  const { on } = useSocket();

  // Socket.IO event listeners
  useEffect(() => {
    // Socket connection status
    const unsubConnect = on('connect', () => {
      setIsSocketConnected(true);
      setStatus('Connected');
    });

    const unsubDisconnect = on('disconnect', () => {
      setIsSocketConnected(false);
      setStatus('Disconnected');
    });

    // Typing indicators
    const unsubTyping = on('typing_indicator', (data: { status: string }) => {
      if (data.status === 'typing') {
        setIsTyping(true);
        setStatus('Eolas is thinking...');
      } else if (data.status === 'done') {
        setIsTyping(false);
        setStatus('Ready');
      } else if (data.status === 'error') {
        setIsTyping(false);
        setStatus('Error occurred');
      }
    });

    // Status updates
    const unsubStatus = on('status_update', (data: { status: string }) => {
      setStatus(data.status);
    });

    // Cleanup
    return () => {
      unsubConnect();
      unsubDisconnect();
      unsubTyping();
      unsubStatus();
    };
  }, [on]);

  useEffect(() => {
    if (messageContainerRef.current) {
      messageContainerRef.current.scrollTop = messageContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-resize textarea as user types
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const sendMessage = async () => {
    if (!input.trim() || isProcessing || !isWalletConnected) return;

    setIsProcessing(true);
    const userMessage = input.trim();
    setInput('');
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    // Add user message to UI
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userMessage, metadata: { timestamp: new Date().toISOString() } },
    ]);

    try {
      setStatus('Processing...');
      
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          use_tools: useTools,
          wallet_address: address,
          model: currentModel
        }),
      });

      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      // Add assistant message to UI
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.text,
          metadata: {
            timestamp: data.timestamp || new Date().toISOString(),
            tool_used: data.tool_used,
            usage: data.usage,
          },
        },
      ]);

      if (data.usage?.total_tokens) {
        setTokenCount(data.usage.total_tokens);
      }
    } catch (error) {
      console.error('Error:', error);
      setStatus(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      
      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          metadata: { timestamp: new Date().toISOString() },
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const toggleTools = () => {
    setUseTools((prev) => !prev);
    setStatus(`Tools ${!useTools ? 'enabled' : 'disabled'}`);
  };

  const clearChat = () => {
    setMessages([
      {
        role: 'system',
        content: "Hello, I'm Eolas. How can I assist you today?",
      },
    ]);
    setTokenCount(0);
    setStatus('Chat cleared');
    fetch('/api/reset', { method: 'POST' });
  };
  
  const changeModel = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCurrentModel(e.target.value);
    setStatus(`Model changed to ${e.target.value}`);
  };

  // Update the marked usage to handle the output synchronously
  const renderMarkdown = (content: string) => {
    return marked.parse(content, {
      highlight: (code, lang) => {
        if (lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return hljs.highlightAuto(code).value;
      },
    }) as string;
  };

  const formatTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-full rounded-xl shadow-2xl overflow-hidden border border-gray-200 bg-white">
      {/* Header */}
      <div className="flex justify-between items-center p-4 bg-white border-b border-gray-200 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="text-2xl bg-purple-600 text-white p-2 rounded-lg">
            <i className="fas fa-robot"></i>
          </div>
          <h1 className="text-xl font-semibold text-gray-800">Eolas</h1>
        </div>
        
        <div className="flex items-center gap-3">
          <ConnectButton />
          
          <div className="flex">
            <button
              onClick={toggleTools}
              className={`px-3 py-2 rounded-l-lg flex items-center gap-2 border ${
                useTools ? 'bg-purple-100 border-purple-300 text-purple-700' : 'bg-gray-50 border-gray-300 text-gray-700'
              }`}
            >
              <i className={`fas fa-wrench ${useTools ? 'text-purple-700' : 'text-gray-500'}`}></i>
              <span className="hidden sm:inline">Tools</span>
            </button>
            
            <div className="border-r border-gray-300 h-8 my-auto mx-1"></div>
            
            <select
              value={currentModel}
              onChange={changeModel}
              className="px-3 py-2 border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-purple-300 text-sm"
            >
              <option value="gpt-4o">GPT-4o</option>
              <option value="gpt-4o-mini">GPT-4o Mini</option>
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
            </select>
            
            <button
              onClick={clearChat}
              className="ml-2 px-3 py-2 rounded-lg border border-gray-300 hover:bg-gray-50 text-gray-700 flex items-center gap-2"
            >
              <i className="fas fa-trash-alt text-gray-600"></i>
              <span className="hidden sm:inline">Clear</span>
            </button>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-gray-50 overflow-hidden">
        {/* Messages */}
        <div
          ref={messageContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-6"
          style={{ backgroundImage: 'url("https://www.transparenttextures.com/patterns/subtle-white-feathers.png")' }}
        >
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role !== 'user' && (
                <div className="w-8 h-8 rounded-full bg-purple-600 text-white flex items-center justify-center mr-2 mt-1 flex-shrink-0">
                  <i className="fas fa-robot text-xs"></i>
                </div>
              )}
              
              <div
                className={`max-w-[80%] ${
                  message.role === 'user'
                    ? 'bg-purple-600 text-white rounded-2xl rounded-tr-none shadow-md'
                    : message.role === 'system'
                    ? 'bg-gray-200 text-gray-800 rounded-2xl border border-gray-300'
                    : 'bg-white text-gray-800 rounded-2xl rounded-tl-none shadow-md border border-gray-200'
                } px-4 py-3`}
              >
                <div
                  className="prose prose-sm max-w-none break-words"
                  dangerouslySetInnerHTML={{
                    __html: renderMarkdown(message.content),
                  }}
                />
                
                {message.metadata?.timestamp && (
                  <div className="flex justify-between items-center mt-2 text-xs opacity-70">
                    <span className={message.role === 'user' ? 'text-purple-100' : 'text-gray-500'}>
                      {formatTime(message.metadata.timestamp)}
                    </span>
                    {message.metadata.tool_used && (
                      <span className="bg-black/10 backdrop-blur-sm text-white px-2 py-1 rounded-full">
                        <i className="fas fa-tools mr-1"></i>
                        {message.metadata.tool_used}
                      </span>
                    )}
                  </div>
                )}
              </div>
              
              {message.role === 'user' && (
                <div className="w-8 h-8 rounded-full bg-gray-400 text-white flex items-center justify-center ml-2 mt-1 flex-shrink-0">
                  <i className="fas fa-user text-xs"></i>
                </div>
              )}
            </div>
          ))}
          
          {/* Typing indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className="w-8 h-8 rounded-full bg-purple-600 text-white flex items-center justify-center mr-2 flex-shrink-0">
                <i className="fas fa-robot text-xs"></i>
              </div>
              <div className="bg-white text-gray-800 rounded-2xl rounded-tl-none shadow-sm border border-gray-200 p-4">
                <div className="flex space-x-2">
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '600ms' }}></div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Status Bar */}
        <div className="py-2 px-4 border-t border-gray-200 bg-white flex justify-between items-center text-xs text-gray-500">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
            <span>{status}</span>
          </div>
          <div className="flex items-center gap-2">
            <i className="fas fa-coins text-yellow-500"></i>
            <span>Tokens: {tokenCount}</span>
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-white border-t border-gray-200">
          <div className="flex rounded-lg border border-gray-300 focus-within:ring-2 focus-within:ring-purple-500 focus-within:border-purple-500 bg-white">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                isWalletConnected
                  ? 'Type your message here...'
                  : 'Please connect your wallet to chat'
              }
              disabled={!isWalletConnected || isProcessing}
              className="flex-1 p-3 bg-transparent rounded-l-lg resize-none max-h-32 focus:outline-none"
              rows={1}
            />
            <button
              onClick={sendMessage}
              disabled={!isWalletConnected || isProcessing || !input.trim()}
              className={`px-4 rounded-r-lg flex items-center justify-center ${
                !isWalletConnected || isProcessing || !input.trim()
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              }`}
            >
              <i className="fas fa-paper-plane"></i>
            </button>
          </div>
          {!isWalletConnected && (
            <div className="mt-2 text-center text-sm text-red-500">
              <i className="fas fa-exclamation-circle mr-1"></i>
              Please connect your wallet to start chatting
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 