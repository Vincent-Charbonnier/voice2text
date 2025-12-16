import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  name: string;
  preferred_username?: string;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Configuration - these would come from environment variables in production
const KEYCLOAK_URL = import.meta.env.VITE_KEYCLOAK_URL || 'https://your-keycloak-server.com';
const KEYCLOAK_REALM = import.meta.env.VITE_KEYCLOAK_REALM || 'your-realm';
const KEYCLOAK_CLIENT_ID = import.meta.env.VITE_KEYCLOAK_CLIENT_ID || 'your-client-id';
const REDIRECT_URI = window.location.origin;

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for token in URL (OAuth callback) or localStorage
  useEffect(() => {
    const checkAuth = async () => {
      // Check URL for OAuth callback
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');

      if (code) {
        // Exchange code for token (would call your backend)
        try {
          // For demo purposes, we'll simulate a successful login
          // In production, you'd exchange the code with your backend
          const mockUser: User = {
            id: 'user-1',
            email: 'user@example.com',
            name: 'Demo User',
            preferred_username: 'demouser',
          };
          setUser(mockUser);
          localStorage.setItem('auth_user', JSON.stringify(mockUser));
          // Clean URL
          window.history.replaceState({}, document.title, window.location.pathname);
        } catch (error) {
          console.error('Failed to exchange code:', error);
        }
      } else {
        // Check localStorage for existing session
        const storedUser = localStorage.getItem('auth_user');
        if (storedUser) {
          setUser(JSON.parse(storedUser));
        }
      }

      setIsLoading(false);
    };

    checkAuth();
  }, []);

  const login = useCallback(() => {
    // Redirect to Keycloak login
    const authUrl = `${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/auth`;
    const params = new URLSearchParams({
      client_id: KEYCLOAK_CLIENT_ID,
      redirect_uri: REDIRECT_URI,
      response_type: 'code',
      scope: 'openid profile email',
    });
    window.location.href = `${authUrl}?${params.toString()}`;
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    localStorage.removeItem('auth_user');
    // Optionally redirect to Keycloak logout
    // const logoutUrl = `${KEYCLOAK_URL}/realms/${KEYCLOAK_REALM}/protocol/openid-connect/logout`;
    // window.location.href = `${logoutUrl}?redirect_uri=${encodeURIComponent(REDIRECT_URI)}`;
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
