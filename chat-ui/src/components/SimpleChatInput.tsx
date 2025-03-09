'use client'

import { useState, useRef, useEffect } from 'react';

interface SimpleChatInputProps {
  onSendMessage: (message: string) => void;
}

const SimpleChatInput: React.FC<SimpleChatInputProps> = ({ onSendMessage }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea as content grows
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(textarea.scrollHeight, 150); // Max height of 150px
      textarea.style.height = `${newHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent);
    }
  };

  return (
    <form 
      style={{
        display: 'flex',
        gap: '12px',
        backgroundColor: '#F1F8E9',
        borderRadius: '16px',
        padding: '12px',
        border: '2px solid #A5D6A7',
        boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
      }} 
      onSubmit={handleSubmit}
    >
      <textarea
        ref={textareaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Tell Pip what you'd like to do..."
        rows={1}
        style={{
          flex: 1,
          border: 'none',
          resize: 'none',
          padding: '10px',
          fontSize: '16px',
          outline: 'none',
          fontFamily: '"Comic Sans MS", "Comic Sans", cursive',
          borderRadius: '10px',
          backgroundColor: 'white',
          color: '#333',
          minHeight: '24px',
          maxHeight: '150px',
          overflowY: 'auto',
          lineHeight: '1.5'
        }}
      />
      <button 
        type="submit" 
        disabled={!message.trim()}
        style={{
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '12px',
          padding: '10px 18px',
          cursor: message.trim() ? 'pointer' : 'not-allowed',
          transition: 'all 0.2s ease',
          opacity: message.trim() ? 1 : 0.6,
          fontWeight: 'bold',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
          minWidth: '80px',
          fontFamily: '"Comic Sans MS", "Comic Sans", cursive'
        }}
      >
        <span>Send</span>
      </button>
    </form>
  );
};

export default SimpleChatInput; 