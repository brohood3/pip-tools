import React from 'react';
import './StatusBar.css';

const TypingIndicator = () => {
  return (
    <div className="typing-indicator">
      <span></span>
      <span></span>
      <span></span>
    </div>
  );
};

const StatusBar = ({ isTyping, status, tokenCount }) => {
  return (
    <div className="status-bar">
      <div className="status-left">
        {isTyping ? <TypingIndicator /> : null}
        <div id="statusText">{status}</div>
      </div>
      <div className="status-right">
        <div id="tokenCounter">Tokens: {tokenCount}</div>
      </div>
    </div>
  );
};

export default StatusBar; 