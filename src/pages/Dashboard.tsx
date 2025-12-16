import React, { useCallback } from 'react';
import { Header } from '@/components/Header';
import { RecordingPanel } from '@/components/RecordingPanel';
import { TranscriptEditor } from '@/components/TranscriptEditor';
import { SummaryPanel } from '@/components/SummaryPanel';
import { useTranscription } from '@/hooks/useTranscription';
import { useSummary } from '@/hooks/useSummary';
import { MediaType } from '@/hooks/useMediaRecorder';
import { useToast } from '@/hooks/use-toast';

const Dashboard: React.FC = () => {
  const { toast } = useToast();
  const {
    transcript,
    isTranscribing,
    error: transcriptionError,
    transcribe,
    updateSegment,
    clearTranscript,
  } = useTranscription();

  const { summary, isGenerating, error: summaryError, generateSummary, clearSummary } = useSummary();

  // Show error toasts
  React.useEffect(() => {
    if (transcriptionError) {
      toast({
        title: 'Transcription Error',
        description: transcriptionError,
        variant: 'destructive',
      });
    }
  }, [transcriptionError, toast]);

  React.useEffect(() => {
    if (summaryError) {
      toast({
        title: 'Summary Error',
        description: summaryError,
        variant: 'destructive',
      });
    }
  }, [summaryError, toast]);

  const handleRecordingComplete = useCallback(
    async (blob: Blob, type: MediaType) => {
      toast({
        title: 'Recording complete',
        description: `Processing your ${type} recording...`,
      });

      clearTranscript();
      clearSummary();
      await transcribe(blob);
    },
    [transcribe, clearTranscript, clearSummary, toast]
  );

  const handleFileUpload = useCallback(
    async (file: File) => {
      toast({
        title: 'File uploaded',
        description: `Processing ${file.name}...`,
      });

      clearTranscript();
      clearSummary();
      await transcribe(file);
    },
    [transcribe, clearTranscript, clearSummary, toast]
  );

  const handleGenerateSummary = useCallback(
    (customPrompt?: string) => {
      generateSummary(transcript, customPrompt);
    },
    [generateSummary, transcript]
  );

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="container py-8">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Left Column - Recording */}
          <div className="space-y-6">
            <RecordingPanel
              onRecordingComplete={handleRecordingComplete}
              onFileUpload={handleFileUpload}
            />
            <SummaryPanel
              transcript={transcript}
              summary={summary}
              isGenerating={isGenerating}
              onGenerateSummary={handleGenerateSummary}
            />
          </div>

          {/* Right Column - Transcript */}
          <div className="lg:col-span-2">
            <TranscriptEditor
              transcript={transcript}
              isLoading={isTranscribing}
              onUpdateSegment={updateSegment}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
