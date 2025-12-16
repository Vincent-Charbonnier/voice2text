import React, { useRef, useEffect } from 'react';
import { useMediaRecorder, MediaType } from '@/hooks/useMediaRecorder';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Mic, Video, Square, Pause, Play, Upload, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface RecordingPanelProps {
  onRecordingComplete: (blob: Blob, type: MediaType) => void;
  onFileUpload: (file: File) => void;
}

export const RecordingPanel: React.FC<RecordingPanelProps> = ({
  onRecordingComplete,
  onFileUpload,
}) => {
  const [mediaType, setMediaType] = React.useState<MediaType>('audio');
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    isRecording,
    isPaused,
    duration,
    startRecording,
    stopRecording,
    pauseRecording,
    resumeRecording,
    stream,
    error,
  } = useMediaRecorder({ onRecordingComplete });

  useEffect(() => {
    if (videoRef.current && stream && mediaType === 'video') {
      videoRef.current.srcObject = stream;
    }
  }, [stream, mediaType]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Record or Upload</CardTitle>
          <Tabs value={mediaType} onValueChange={(v) => setMediaType(v as MediaType)}>
            <TabsList className="h-8">
              <TabsTrigger value="audio" className="text-xs px-3" disabled={isRecording}>
                <Mic className="h-3.5 w-3.5 mr-1.5" />
                Audio
              </TabsTrigger>
              <TabsTrigger value="video" className="text-xs px-3" disabled={isRecording}>
                <Video className="h-3.5 w-3.5 mr-1.5" />
                Video
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Preview Area */}
        <div
          className={cn(
            'relative flex items-center justify-center rounded-lg bg-muted/50 overflow-hidden transition-all',
            mediaType === 'video' ? 'aspect-video' : 'h-32'
          )}
        >
          {mediaType === 'video' && isRecording ? (
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="flex flex-col items-center gap-3">
              {isRecording ? (
                <>
                  {/* Audio Waveform Animation */}
                  <div className="flex items-end gap-1 h-8">
                    {[...Array(5)].map((_, i) => (
                      <div
                        key={i}
                        className={cn(
                          'w-1 bg-primary rounded-full transition-all',
                          `animate-wave-delay-${i}`
                        )}
                        style={{ height: '4px' }}
                      />
                    ))}
                  </div>
                  <span className="text-sm text-muted-foreground">
                    {isPaused ? 'Paused' : 'Recording...'}
                  </span>
                </>
              ) : (
                <>
                  {mediaType === 'audio' ? (
                    <Mic className="h-8 w-8 text-muted-foreground/50" />
                  ) : (
                    <Video className="h-8 w-8 text-muted-foreground/50" />
                  )}
                  <span className="text-sm text-muted-foreground">
                    Click record to start
                  </span>
                </>
              )}
            </div>
          )}

          {/* Recording Indicator */}
          {isRecording && (
            <div className="absolute top-3 right-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-recording/10 border border-recording/20">
              <div className={cn(
                'h-2 w-2 rounded-full bg-recording',
                !isPaused && 'animate-pulse-recording'
              )} />
              <span className="text-xs font-medium text-recording">
                {formatDuration(duration)}
              </span>
            </div>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
            <AlertCircle className="h-4 w-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {/* Controls */}
        <div className="flex items-center justify-center gap-3">
          {!isRecording ? (
            <>
              <Button
                size="lg"
                onClick={() => startRecording(mediaType)}
                className="h-14 w-14 rounded-full"
              >
                {mediaType === 'audio' ? (
                  <Mic className="h-6 w-6" />
                ) : (
                  <Video className="h-6 w-6" />
                )}
              </Button>
              <div className="h-8 w-px bg-border" />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="gap-2"
              >
                <Upload className="h-4 w-4" />
                Upload File
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </>
          ) : (
            <>
              <Button
                variant="outline"
                size="icon"
                onClick={isPaused ? resumeRecording : pauseRecording}
                className="h-12 w-12 rounded-full"
              >
                {isPaused ? (
                  <Play className="h-5 w-5" />
                ) : (
                  <Pause className="h-5 w-5" />
                )}
              </Button>
              <Button
                variant="destructive"
                size="lg"
                onClick={stopRecording}
                className="h-14 w-14 rounded-full"
              >
                <Square className="h-5 w-5" />
              </Button>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
