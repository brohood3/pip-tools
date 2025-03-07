import axios from 'axios';
import io from 'socket.io-client';

// API base URL - adjust this based on your backend setup
const API_BASE_URL = 'http://localhost:8080/api';

// Setup Socket.io connection
let socket;

const initializeSocket = (callbacks) => {
  socket = io('http://localhost:8080', {
    reconnectionAttempts: 5,
    reconnectionDelay: 1000,
  });

  // Connection events
  socket.on('connect', () => {
    console.log('Socket connected');
    if (callbacks?.onConnect) callbacks.onConnect();
  });

  socket.on('disconnect', () => {
    console.log('Socket disconnected');
    if (callbacks?.onDisconnect) callbacks.onDisconnect();
  });

  // Chat-specific events
  socket.on('typing_indicator', (data) => {
    console.log('Typing indicator:', data);
    if (callbacks?.onTypingIndicator) callbacks.onTypingIndicator(data);
  });

  socket.on('status_update', (data) => {
    console.log('Status update:', data);
    if (callbacks?.onStatusUpdate) callbacks.onStatusUpdate(data);
  });

  return socket;
};

const disconnectSocket = () => {
  if (socket) {
    socket.disconnect();
  }
};

// API functions

// Send a message to the chat API
const sendMessage = async (message, useTools = true) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/chat`, {
      message,
      use_tools: useTools
    });
    return response.data;
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

// Get available models
const getModels = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/models`);
    return response.data;
  } catch (error) {
    console.error('Error getting models:', error);
    throw error;
  }
};

// Set the current model
const setModel = async (model) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/models`, { model });
    return response.data;
  } catch (error) {
    console.error('Error setting model:', error);
    throw error;
  }
};

// Get token usage statistics
const getUsage = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/usage`);
    return response.data;
  } catch (error) {
    console.error('Error getting usage:', error);
    throw error;
  }
};

// Reset token usage statistics
const resetUsage = async () => {
  try {
    const response = await axios.post(`${API_BASE_URL}/reset`);
    return response.data;
  } catch (error) {
    console.error('Error resetting usage:', error);
    throw error;
  }
};

export {
  initializeSocket,
  disconnectSocket,
  sendMessage,
  getModels,
  setModel,
  getUsage,
  resetUsage
}; 