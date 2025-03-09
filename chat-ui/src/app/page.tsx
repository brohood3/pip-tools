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
    { role: 'assistant', content: "Hello! I'm Pip, a baby AI just learning about the world! 👶 I have some fun toys (tools) that I'm learning to play with. Can I help you with something today? I might make some mistakes, but I'm excited to try!" }
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
      { role: 'assistant', content: "Hello! I'm Pip, a baby AI just learning about the world! 👶 I have some fun toys (tools) that I'm learning to play with. Can I help you with something today? I might make some mistakes, but I'm excited to try!" }
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
          <div style={{ 
            fontSize: '2.5rem', 
            color: '#4CAF50', 
            backgroundColor: '#E8F5E9',
            borderRadius: '50%',
            width: '60px',
            height: '60px',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
          }}>
            <i className="fab fa-android"></i>
          </div>
          <h1 style={{ 
            margin: 0, 
            fontSize: '2.2rem', 
            fontWeight: '600',
            color: '#4CAF50',
            fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
          }}>Pip</h1>
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
            backgroundColor: '#E8F5E9',
            border: '1px solid #A5D6A7',
            borderRadius: '12px',
            padding: '0 8px'
          }}>
            <i className="fas fa-brain" style={{ color: '#4CAF50' }}></i>
            <select 
              value={model}
              onChange={(e) => setModel(e.target.value)}
              style={{
                border: 'none',
                background: 'transparent',
                padding: '10px 0',
                fontSize: '14px',
                color: '#333',
                cursor: 'pointer',
                fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
              }}
            >
              <option value="gpt-4o">Smart Brain</option>
              <option value="gpt-4o-mini">Medium Brain</option>
              <option value="gpt-3.5-turbo">Small Brain</option>
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
              backgroundColor: useTools ? '#4CAF50' : '#f0f0f0',
              color: useTools ? 'white' : 'black',
              border: 'none',
              borderRadius: '12px',
              cursor: 'pointer',
              fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
            }}
          >
            <i className="fas fa-puzzle-piece"></i>
            <span>Toys: {useTools ? 'On' : 'Off'}</span>
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
              borderRadius: '12px',
              cursor: 'pointer',
              fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
            }}
          >
            <i className="fas fa-eraser"></i>
            <span>Start Over</span>
          </button>
        </div>
        
        {/* Chat UI */}
        <div className="chat-container" style={{ 
          width: '100%',
          border: '2px solid #A5D6A7', 
          borderRadius: '16px', 
          padding: '16px', 
          boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
          backgroundColor: '#FAFFF9'
        }}>
          <div className="message-container" style={{ 
            height: '450px', 
            overflowY: 'auto', 
            marginBottom: '16px',
            padding: '8px',
            backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'100\' height=\'100\' viewBox=\'0 0 100 100\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cpath d=\'M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z\' fill=\'%234caf50\' fill-opacity=\'0.05\' fill-rule=\'evenodd\'/%3E%3C/svg%3E")',
            backgroundSize: '150px 150px'
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
                    backgroundColor: message.role === 'user' ? '#2196F3' : '#4CAF50',
                    color: 'white',
                    marginRight: message.role === 'user' ? '0' : '12px',
                    marginLeft: message.role === 'user' ? '12px' : '0',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                    fontSize: message.role === 'assistant' ? '18px' : '16px'
                  }}
                >
                  {message.role === 'user' ? <i className="fas fa-user"></i> : <i className="fab fa-android"></i>}
                </div>
                <div 
                  className="message-content"
                  style={{ 
                    padding: '12px',
                    borderRadius: '16px',
                    backgroundColor: message.role === 'user' ? '#E3F2FD' : '#E8F5E9',
                    color: '#333',
                    border: message.role === 'assistant' ? '1px solid #A5D6A7' : '1px solid #BBDEFB',
                    maxWidth: '75%',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                    textAlign: 'left',
                    fontFamily: message.role === 'assistant' ? '"Comic Sans MS", "Comic Sans", cursive' : 'inherit'
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
                      backgroundColor: '#F1F8E9',
                      padding: '10px',
                      borderRadius: '12px',
                      border: '1px dashed #A5D6A7',
                      color: '#333',
                      fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                        <i className="fas fa-puzzle-piece" style={{ color: '#4CAF50' }}></i>
                        <span style={{ fontWeight: 'bold' }}>Toy: {message.toolUsed}</span>
                      </div>
                      
                      {/* Transaction button for brian_transaction tool */}
                      {message.toolUsed === 'brian_transaction' && message.transaction_data && (
                        <div style={{ marginTop: '12px' }}>
                          <button
                            onClick={() => handleExecuteTransaction(message.transaction_data)}
                            style={{
                              backgroundColor: '#4CAF50',
                              color: 'white',
                              border: 'none',
                              borderRadius: '10px',
                              padding: '8px 14px',
                              fontSize: '13px',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: '6px',
                              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                              transition: 'all 0.2s ease',
                              fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
                            }}
                          >
                            <i className="fas fa-wallet"></i>
                            <span>Use Wallet</span>
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