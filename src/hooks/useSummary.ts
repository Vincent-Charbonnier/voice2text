import { useState, useCallback } from 'react';
import { TranscriptSegment } from './useTranscription';
import { useApiConfig } from './useApiConfig';

interface UseSummaryReturn {
  summary: string;
  isGenerating: boolean;
  error: string | null;
  generateSummary: (transcript: TranscriptSegment[], customPrompt?: string) => Promise<void>;
  clearSummary: () => void;
}

export const useSummary = (): UseSummaryReturn => {
  const [summary, setSummary] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { summaryEndpoint, getAuthHeaders } = useApiConfig();

  const generateSummary = useCallback(async (transcript: TranscriptSegment[], customPrompt?: string) => {
    setIsGenerating(true);
    setError(null);

    try {
      const fullText = transcript.map((seg) => seg.text).join(' ');

      const response = await fetch(summaryEndpoint, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          text: fullText,
          prompt: customPrompt,
          segments: transcript, // Send full segments for context
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Summary generation failed: ${response.statusText}`);
      }

      const data = await response.json();
      setSummary(data.summary || data.text || '');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Summary generation failed';
      setError(message);
      console.error('Summary error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [summaryEndpoint, getAuthHeaders]);

  const clearSummary = useCallback(() => {
    setSummary('');
    setError(null);
  }, []);

  return {
    summary,
    isGenerating,
    error,
    generateSummary,
    clearSummary,
  };
};
