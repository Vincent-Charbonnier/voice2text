import { useAuth } from '@/contexts/AuthContext';

interface ApiConfig {
  baseUrl: string;
  transcriptionEndpoint: string;
  summaryEndpoint: string;
  wsTranscriptionEndpoint: string;
  getAuthHeaders: () => HeadersInit;
}

export const useApiConfig = (): ApiConfig => {
  const { user } = useAuth();

  const baseUrl = import.meta.env.VITE_API_BASE_URL || '';
  const transcriptionEndpoint = import.meta.env.VITE_TRANSCRIPTION_API_URL || `${baseUrl}/api/transcribe`;
  const summaryEndpoint = import.meta.env.VITE_SUMMARY_API_URL || `${baseUrl}/api/summarize`;
  
  // WebSocket endpoint for streaming transcription
  const wsBase = baseUrl.replace(/^http/, 'ws');
  const wsTranscriptionEndpoint = import.meta.env.VITE_WS_TRANSCRIPTION_URL || `${wsBase}/ws/transcribe`;

  const getAuthHeaders = (): HeadersInit => {
    const token = localStorage.getItem('auth_token');
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    return headers;
  };

  return {
    baseUrl,
    transcriptionEndpoint,
    summaryEndpoint,
    wsTranscriptionEndpoint,
    getAuthHeaders,
  };
};
