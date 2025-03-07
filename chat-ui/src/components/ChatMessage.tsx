'use client'

import React from 'react';
import { marked } from 'marked';

interface ChatMessageProps {
  role: string;
  content: string;
  timestamp?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ role, content, timestamp }) => {
  const isUser = role === 'user';

  // Parse markdown in assistant messages
  const processedContent = isUser 
    ? content 
    : marked.parse(content) as string;

  return (
    <div className={`message ${role}`}>
      <div className={`avatar ${role}-avatar`}>
        {isUser ? (
          <i className="fas fa-user"></i>
        ) : (
          <i className="fas fa-robot"></i>
        )}
      </div>
      <div className="message-content">
        {isUser ? (
          <p>{content}</p>
        ) : (
          <div dangerouslySetInnerHTML={{ __html: processedContent }} />
        )}
        {timestamp && (
          <div className="message-timestamp">
            {new Date(timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage; 