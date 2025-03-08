'use client'

import { ConnectButton } from "@/components/ConnectButton";
import { useState, useRef, useEffect } from 'react';
import SimpleChatInput from '@/components/SimpleChatInput';
import { useAccount, useSendTransaction, useWalletClient } from 'wagmi';
import { modal } from '@/context';
import { useAppKit } from '@reown/appkit/react';
import { useConnect, useDisconnect, useWriteContract } from 'wagmi';
import { injected } from 'wagmi/connectors';
import { parseEther } from 'viem';
import { FaRobot } from 'react-icons/fa';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

// Define the message type
interface Message {
  role: string;
  content: string;
  toolUsed?: string | null;
  timestamp?: string;
  confidence?: string;
  reasoning?: string;
  transaction_data?: any;
  chart?: string;
}

export default function Home() {
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: "Hey there! I'm Nova, an AI with a flair for the dramatic and a passion for problem-solving. What's on your mind today? 💫" }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [model, setModel] = useState("gpt-4o");
  const [useTools, setUseTools] = useState(true);
  const [tokenUsage, setTokenUsage] = useState({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  // Get wallet connection state
  const { address, isConnected } = useAccount();
  const { open } = useAppKit();
  const { data: walletClient } = useWalletClient();
  const { sendTransactionAsync } = useSendTransaction();

  // Scroll to bottom whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle executing a transaction
  const handleExecuteTransaction = async (transactionData: any) => {
    try {
      // Add detailed logging to understand the structure
      console.log("Raw transaction data:", JSON.stringify(transactionData, null, 2));
      
      if (!isConnected || !address) {
        // Open wallet modal if not connected
        open();
        return;
      }
      
      // Check if we have valid transaction data
      if (!transactionData) {
        console.error("Transaction data is null or undefined");
        alert("Transaction data is missing.");
        return;
      }
      
      // Try different possible structures for the transaction data
      let txStep = null;
      
      // Option 1: Direct transaction data
      if (transactionData.to && transactionData.data) {
        txStep = transactionData;
        console.log("Using direct transaction data");
      } 
      // Option 2: Nested in result[0].data.steps[0]
      else if (
        transactionData.result && 
        Array.isArray(transactionData.result) && 
        transactionData.result.length > 0 &&
        transactionData.result[0].data &&
        transactionData.result[0].data.steps &&
        Array.isArray(transactionData.result[0].data.steps) &&
        transactionData.result[0].data.steps.length > 0
      ) {
        txStep = transactionData.result[0].data.steps[0];
        console.log("Using nested transaction data from result[0].data.steps[0]");
      }
      // Option 3: Try to find any object with to, value, and data properties
      else {
        const findTransactionStep = (obj: any): any => {
          if (!obj || typeof obj !== 'object') return null;
          
          // Check if this object has the required properties
          if (obj.to && (obj.data || obj.data === '0x')) {
            return obj;
          }
          
          // Check all properties recursively
          for (const key in obj) {
            if (typeof obj[key] === 'object') {
              const result = findTransactionStep(obj[key]);
              if (result) return result;
            }
          }
          
          return null;
        };
        
        txStep = findTransactionStep(transactionData);
        console.log("Using transaction data found by recursive search");
      }
      
      // If we still don't have transaction data, show an error
      if (!txStep) {
        console.error("Could not find valid transaction data in the response");
        console.error("Transaction data structure:", transactionData);
        alert("Transaction data is invalid or incomplete. Could not find transaction details.");
        return;
      }
      
      console.log("Extracted transaction step:", txStep);
      
      // Extract the transaction parameters
      const to = txStep.to;
      const value = txStep.value || '0';
      const data = txStep.data || '0x';
      
      if (!to) {
        console.error("Transaction missing 'to' address");
        alert("Transaction data is incomplete. Missing destination address.");
        return;
      }
      
      // Prepare transaction parameters
      const transaction = {
        to,
        value: BigInt(value),
        data,
      };
      
      console.log("Sending transaction:", transaction);
      
      // Send the transaction
      const hash = await sendTransactionAsync(transaction);
      
      console.log("Transaction sent:", hash);
      
      // Add a message to the chat about the transaction
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Transaction sent! Transaction hash: ${hash}`
      }]);
      
    } catch (error) {
      console.error("Error executing transaction:", error);
      
      // Add error message to chat
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error executing transaction: ${error instanceof Error ? error.message : String(error)}`
      }]);
    }
  };

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
        reasoning: data.reasoning,
        transaction_data: data.transaction_data || null,
        chart: data.chart || null
      }]);
      
      // If there's transaction data, log it to the console
      if (data.transaction_data) {
        console.log("Transaction data received:", data.transaction_data);
        // Here you would trigger your wallet modal or transaction handling
        // For now, we'll just log it to the console
      }
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
      { role: 'assistant', content: "Hey there! I'm Nova, an AI with a flair for the dramatic and a passion for problem-solving. What's on your mind today? 💫" }
    ]);
  };

  return (
    <div className="pages">
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        maxWidth: '900px', 
        margin: '0 auto', 
        padding: '20px'
      }}>
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '15px', 
          marginBottom: '20px'
        }}>
          <div style={{ fontSize: '2.5rem', color: '#6e48aa' }}>
            <i className="fas fa-robot"></i>
          </div>
          <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: '600' }}>AI Chat</h1>
        </div>

        {/* Header controls */}
        <div style={{ 
          display: 'flex', 
          flexWrap: 'wrap', 
          justifyContent: 'center', 
          gap: '10px', 
          marginBottom: '20px',
          width: '100%'
        }}>
          {/* Wallet connection */}
          <ConnectButton />
          
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
                fontSize: '14px',
                color: '#333',
                cursor: 'pointer'
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
        </div>
        
        {/* Chat UI */}
        <div className="chat-container" style={{ 
          width: '100%',
          border: '1px solid #e0e0e0', 
          borderRadius: '12px', 
          padding: '16px', 
          boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
          backgroundColor: '#fcfcfc'
        }}>
          <div className="message-container" style={{ 
            height: '450px', 
            overflowY: 'auto', 
            marginBottom: '16px',
            padding: '8px'
          }}>
            {messages.map((message, index) => (
              <div 
                key={index} 
                className={`message ${message.role}`}
                style={{ 
                  display: 'flex', 
                  margin: '12px 0',
                  flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
                  alignItems: 'flex-start'
                }}
              >
                <div 
                  className={`avatar ${message.role}-avatar`}
                  style={{ 
                    width: '38px', 
                    height: '38px', 
                    borderRadius: '50%', 
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: message.role === 'user' ? '#007bff' : '#6e48aa',
                    color: 'white',
                    marginRight: message.role === 'user' ? '0' : '12px',
                    marginLeft: message.role === 'user' ? '12px' : '0',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                  }}
                >
                  {message.role === 'user' ? 'U' : 'AI'}
                </div>
                <div 
                  className="message-content"
                  style={{ 
                    padding: '12px',
                    borderRadius: '12px',
                    backgroundColor: message.role === 'user' ? '#007bff' : 'white',
                    color: message.role === 'user' ? 'white' : '#333',
                    border: message.role === 'assistant' ? '1px solid #eaeaea' : 'none',
                    maxWidth: '75%',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                    textAlign: 'left'
                  }}
                >
                  <div className="markdown-content" style={{ overflow: 'hidden' }}>
                    {message.role === 'user' ? (
                      // For user messages, just use simple text with line breaks
                      message.content.split('\n').map((text, i) => (
                        <p key={i} style={{ margin: '0 0 8px 0', lineHeight: '1.5' }}>{text}</p>
                      ))
                    ) : (
                      // For assistant messages, use Markdown rendering
                      <ReactMarkdown
                        components={{
                          code({className, children, ...props}: any) {
                            const match = /language-(\w+)/.exec(className || '');
                            return className?.includes('language-') ? (
                              <SyntaxHighlighter
                                style={atomDark}
                                language={match ? match[1] : ''}
                                PreTag="div"
                              >
                                {String(children).replace(/\n$/, '')}
                              </SyntaxHighlighter>
                            ) : (
                              <code className={className} {...props}>
                                {children}
                              </code>
                            );
                          },
                          p: ({children}: any) => <p style={{margin: '0 0 8px 0', lineHeight: '1.5'}}>{children}</p>,
                          ul: ({children}: any) => <ul style={{margin: '0 0 8px 0', paddingLeft: '20px'}}>{children}</ul>,
                          ol: ({children}: any) => <ol style={{margin: '0 0 8px 0', paddingLeft: '20px'}}>{children}</ol>,
                          li: ({children}: any) => <li style={{margin: '4px 0'}}>{children}</li>,
                          h1: ({children}: any) => <h1 style={{margin: '16px 0 8px 0', fontSize: '1.5em'}}>{children}</h1>,
                          h2: ({children}: any) => <h2 style={{margin: '14px 0 8px 0', fontSize: '1.3em'}}>{children}</h2>,
                          h3: ({children}: any) => <h3 style={{margin: '12px 0 8px 0', fontSize: '1.2em'}}>{children}</h3>,
                          blockquote: ({children}: any) => (
                            <blockquote style={{
                              borderLeft: '3px solid #ccc',
                              margin: '8px 0',
                              paddingLeft: '12px',
                              color: '#666'
                            }}>
                              {children}
                            </blockquote>
                          ),
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    )}
                  </div>
                  
                  {message.toolUsed && (
                    <div style={{ 
                      fontSize: '12px', 
                      marginTop: '8px', 
                      backgroundColor: '#f8f0ff',
                      padding: '10px',
                      borderRadius: '8px',
                      border: '1px solid #e0d0ff',
                      color: '#333'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                        <i className="fas fa-tools"></i>
                        <span style={{ fontWeight: 'bold' }}>Tool: {message.toolUsed}</span>
                        {message.confidence && (
                          <span style={{ 
                            marginLeft: 'auto', 
                            backgroundColor: 
                              message.confidence === 'high' ? '#c8e6c9' : 
                              message.confidence === 'medium' ? '#fff9c4' : '#ffcdd2',
                            padding: '3px 8px',
                            borderRadius: '4px',
                            fontSize: '10px',
                            fontWeight: 'bold',
                            color: '#333'
                          }}>
                            {message.confidence.toUpperCase()}
                          </span>
                        )}
                      </div>
                      {message.reasoning && (
                        <div style={{ fontSize: '11px', marginTop: '4px' }}>
                          <span style={{ fontStyle: 'italic' }}>Reasoning: {message.reasoning}</span>
                        </div>
                      )}
                      
                      {/* Transaction button for brian_transaction tool */}
                      {message.toolUsed === 'brian_transaction' && message.transaction_data && (
                        <div style={{ marginTop: '12px' }}>
                          <button
                            onClick={() => handleExecuteTransaction(message.transaction_data)}
                            style={{
                              backgroundColor: '#6e48aa',
                              color: 'white',
                              border: 'none',
                              borderRadius: '6px',
                              padding: '8px 14px',
                              fontSize: '13px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                              transition: 'all 0.2s ease'
                            }}
                          >
                            <i className="fas fa-wallet"></i>
                            <span>Execute Transaction</span>
                          </button>
                        </div>
                      )}
                      
                      {/* Chart display for technical_analysis tool */}
                      {message.toolUsed === 'technical_analysis' && message.chart && (
                        <div style={{ marginTop: '12px', textAlign: 'center' }}>
                          <img 
                            src={`data:image/png;base64,${message.chart}`} 
                            alt="Technical Analysis Chart" 
                            style={{ 
                              maxWidth: '100%', 
                              borderRadius: '8px', 
                              border: '1px solid #ddd',
                              boxShadow: '0 3px 6px rgba(0,0,0,0.1)'
                            }} 
                          />
                        </div>
                      )}
                    </div>
                  )}
                  
                  {message.timestamp && (
                    <div style={{ fontSize: '11px', marginTop: '6px', opacity: 0.7, textAlign: 'right' }}>
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
          <div style={{ fontSize: '12px', textAlign: 'right', color: '#888', marginTop: '8px' }}>
            Tokens: {tokenUsage.total_tokens} (Prompt: {tokenUsage.prompt_tokens}, Completion: {tokenUsage.completion_tokens})
          </div>
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