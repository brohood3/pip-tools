'use client'

import { useState } from 'react';

interface SimpleChatInputProps {
  onSendMessage: (message: string) => void;
}

const SimpleChatInput: React.FC<SimpleChatInputProps> = ({ onSendMessage }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <form 
      style={{
        display: 'flex',
        gap: '10px',
        backgroundColor: 'white',
        borderRadius: '8px',
        padding: '10px',
        border: '1px solid #e0e0e0'
      }} 
      onSubmit={handleSubmit}
    >
      <textarea
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Type your message here..."
        rows={1}
        style={{
          flex: 1,
          border: 'none',
          resize: 'none',
          padding: '8px',
          fontSize: '16px',
          outline: 'none',
          fontFamily: 'inherit'
        }}
      />
      <button 
        type="submit" 
        disabled={!message.trim()}
        style={{
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '8px',
          padding: '8px 16px',
          cursor: 'pointer',
          transition: 'background-color 0.2s',
          opacity: message.trim() ? 1 : 0.6
        }}
      >
        Send
      </button>
    </form>
  );
};

export default SimpleChatInput; 