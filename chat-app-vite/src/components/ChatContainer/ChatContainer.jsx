import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Prism from 'prismjs';
import 'prismjs/themes/prism-okaidia.css';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-css';
import 'prismjs/components/prism-jsx';
import './ChatContainer.css';

const Message = ({ message }) => {
  const { role, content, timestamp, tool_used } = message;
  
  // Apply syntax highlighting after render
  useEffect(() => {
    Prism.highlightAll();
  }, [content]);
  
  return (
    <div className={`message ${role}`}>
      <div className="message-content">
        <ReactMarkdown
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || '');
              return !inline && match ? (
                <pre className={`language-${match[1]}`}>
                  <code className={`language-${match[1]}`} {...props}>
                    {children}
                  </code>
                </pre>
              ) : (
                <code className={className} {...props}>
                  {children}
                </code>
              );
            }
          }}
        >
          {content}
        </ReactMarkdown>
      </div>
      {(timestamp || tool_used) && (
        <div className="message-metadata">
          {timestamp && (
            <span className="message-time">
              {new Date(timestamp).toLocaleTimeString()}
            </span>
          )}
          {tool_used && (
            <span className="tool-badge">
              <i className="fa-solid fa-wrench"></i> {tool_used}
            </span>
          )}
        </div>
      )}
    </div>
  );
};

const ChatContainer = ({ messages }) => {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="message-container">
      {messages.map((message, index) => (
        <Message key={index} message={message} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default ChatContainer; 