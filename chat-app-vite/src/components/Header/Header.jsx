import React from 'react';
import './Header.css';

const Header = ({ onClearChat, onToggleTools, onModelChange, useTools, currentModel }) => {
  return (
    <header className="app-header">
      <div className="logo">
        <i className="fa-solid fa-robot"></i>
        <h1>Eolas</h1>
      </div>
      <div className="header-controls">
        <button 
          className="header-btn" 
          onClick={onToggleTools}
        >
          <i className="fa-solid fa-wrench"></i>
          <span>Tools: {useTools ? 'On' : 'Off'}</span>
        </button>
        <div className="model-selector">
          <i className="fa-solid fa-microchip"></i>
          <select 
            value={currentModel}
            onChange={(e) => onModelChange(e.target.value)}
          >
            <option value="gpt-4o">GPT-4o</option>
            <option value="gpt-4o-mini">GPT-4o Mini</option>
            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
          </select>
        </div>
        <button 
          className="header-btn"
          onClick={onClearChat}
        >
          <i className="fa-solid fa-trash"></i>
          <span>Clear</span>
        </button>
      </div>
    </header>
  );
};

export default Header; 