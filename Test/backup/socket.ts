import { io, Socket } from 'socket.io-client';

// Singleton pattern to manage socket connection
class SocketManager {
  private static instance: SocketManager;
  private socket: Socket | null = null;
  private listeners: Record<string, Array<(data: any) => void>> = {};

  private constructor() {
    // Private constructor to enforce singleton
  }

  public static getInstance(): SocketManager {
    if (!SocketManager.instance) {
      SocketManager.instance = new SocketManager();
    }
    return SocketManager.instance;
  }

  public connect(): Socket {
    if (!this.socket) {
      // Connect to the same origin
      this.socket = io();

      // Add listeners for standard events
      this.socket.on('connect', () => {
        console.log('Socket.IO connected');
        this.notifyListeners('connect', {});
      });

      this.socket.on('disconnect', () => {
        console.log('Socket.IO disconnected');
        this.notifyListeners('disconnect', {});
      });

      this.socket.on('typing_indicator', (data) => {
        this.notifyListeners('typing_indicator', data);
      });

      this.socket.on('status_update', (data) => {
        this.notifyListeners('status_update', data);
      });
    }

    return this.socket;
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  public on(event: string, callback: (data: any) => void): () => void {
    if (!this.listeners[event]) {
      this.listeners[event] = [];
    }
    
    this.listeners[event].push(callback);
    
    // Return a function to remove this specific listener
    return () => {
      if (this.listeners[event]) {
        this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
      }
    };
  }

  private notifyListeners(event: string, data: any): void {
    if (this.listeners[event]) {
      this.listeners[event].forEach(callback => callback(data));
    }
  }
}

// Export singleton instance
export const socketManager = SocketManager.getInstance();

// Helper hook for React components
export function useSocket() {
  return {
    socket: socketManager.connect(),
    on: socketManager.on.bind(socketManager),
    disconnect: socketManager.disconnect.bind(socketManager)
  };
} 