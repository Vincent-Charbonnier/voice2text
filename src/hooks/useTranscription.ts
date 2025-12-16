import { useState, useCallback, useRef } from 'react';
import { useApiConfig } from './useApiConfig';

export interface TranscriptSegment {
  id: string;
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

interface UseTranscriptionReturn {
  transcript: TranscriptSegment[];
  isTranscribing: boolean;
  isStreaming: boolean;
  error: string | null;
  progress: number;
  transcribe: (blob: Blob) => Promise<void>;
  transcribeStream: (stream: MediaStream) => void;
  stopStreaming: () => void;
  updateSegment: (id: string, text: string) => void;
  clearTranscript: () => void;
}

export const useTranscription = (): UseTranscriptionReturn => {
  const [transcript, setTranscript] = useState<TranscriptSegment[]>([]);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  const { transcriptionEndpoint, wsTranscriptionEndpoint, getAuthHeaders } = useApiConfig();
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  // REST API transcription (for file uploads and completed recordings)
  const transcribe = useCallback(async (blob: Blob) => {
    setIsTranscribing(true);
    setError(null);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');

      const headers = getAuthHeaders();
      // Remove Content-Type for FormData (browser sets it with boundary)
      delete (headers as Record<string, string>)['Content-Type'];

      const response = await fetch(transcriptionEndpoint, {
        method: 'POST',
        body: formData,
        headers,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Transcription failed: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Transform response to our format
      const segments: TranscriptSegment[] = data.segments?.map((seg: any, index: number) => ({
        id: `seg-${index}-${Date.now()}`,
        start: seg.start || 0,
        end: seg.end || 0,
        text: seg.text || '',
        speaker: seg.speaker,
      })) || [{
        id: `seg-0-${Date.now()}`,
        start: 0,
        end: 0,
        text: data.text || data.transcript || '',
      }];

      setTranscript(segments);
      setProgress(100);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Transcription failed';
      setError(message);
      console.error('Transcription error:', err);
    } finally {
      setIsTranscribing(false);
    }
  }, [transcriptionEndpoint, getAuthHeaders]);

  // WebSocket streaming transcription (for live recording)
  const transcribeStream = useCallback((stream: MediaStream) => {
    setIsStreaming(true);
    setError(null);
    setTranscript([]);

    try {
      const token = localStorage.getItem('auth_token');
      const wsUrl = token 
        ? `${wsTranscriptionEndpoint}?token=${encodeURIComponent(token)}`
        : wsTranscriptionEndpoint;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected for streaming transcription');
        
        // Start sending audio chunks
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm;codecs=opus') 
            ? 'audio/webm;codecs=opus' 
            : 'audio/webm',
        });

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0 && ws.readyState === WebSocket.OPEN) {
            ws.send(event.data);
          }
        };

        mediaRecorder.start(250); // Send chunks every 250ms
        mediaRecorderRef.current = mediaRecorder;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'partial') {
            // Update the last segment with partial results
            setTranscript((prev) => {
              const lastIndex = prev.length - 1;
              if (lastIndex >= 0 && prev[lastIndex].id.startsWith('partial-')) {
                return prev.map((seg, i) => 
                  i === lastIndex ? { ...seg, text: data.text } : seg
                );
              }
              return [...prev, {
                id: `partial-${Date.now()}`,
                start: data.start || 0,
                end: data.end || 0,
                text: data.text,
                speaker: data.speaker,
              }];
            });
          } else if (data.type === 'final') {
            // Replace partial with final segment
            setTranscript((prev) => {
              const filtered = prev.filter((seg) => !seg.id.startsWith('partial-'));
              return [...filtered, {
                id: `seg-${filtered.length}-${Date.now()}`,
                start: data.start || 0,
                end: data.end || 0,
                text: data.text,
                speaker: data.speaker,
              }];
            });
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Streaming connection error');
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setIsStreaming(false);
      };
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start streaming';
      setError(message);
      setIsStreaming(false);
    }
  }, [wsTranscriptionEndpoint]);

  const stopStreaming = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setIsStreaming(false);
  }, []);

  const updateSegment = useCallback((id: string, text: string) => {
    setTranscript((prev) =>
      prev.map((seg) => (seg.id === id ? { ...seg, text } : seg))
    );
  }, []);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    setError(null);
    setProgress(0);
  }, []);

  return {
    transcript,
    isTranscribing,
    isStreaming,
    error,
    progress,
    transcribe,
    transcribeStream,
    stopStreaming,
    updateSegment,
    clearTranscript,
  };
};
