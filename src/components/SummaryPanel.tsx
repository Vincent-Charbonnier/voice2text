import React, { useState } from 'react';
import { TranscriptSegment } from '@/hooks/useTranscription';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible';
import { Sparkles, ChevronDown, Copy, RefreshCw } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface SummaryPanelProps {
  transcript: TranscriptSegment[];
  summary: string;
  isGenerating: boolean;
  onGenerateSummary: (customPrompt?: string) => void;
}

export const SummaryPanel: React.FC<SummaryPanelProps> = ({
  transcript,
  summary,
  isGenerating,
  onGenerateSummary,
}) => {
  const { toast } = useToast();
  const [customPrompt, setCustomPrompt] = useState('');
  const [isPromptOpen, setIsPromptOpen] = useState(false);

  const handleGenerate = () => {
    onGenerateSummary(customPrompt.trim() || undefined);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
    toast({
      title: 'Copied to clipboard',
      description: 'The summary has been copied.',
    });
  };

  const hasTranscript = transcript.length > 0;

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Sparkles className="h-5 w-5 text-accent" />
          AI Summary
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Custom Prompt Section */}
        <Collapsible open={isPromptOpen} onOpenChange={setIsPromptOpen}>
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              className="w-full justify-between h-auto py-2 px-3 text-sm font-normal"
            >
              <span className="text-muted-foreground">
                {customPrompt ? 'Custom prompt applied' : 'Add custom instructions (optional)'}
              </span>
              <ChevronDown
                className={cn(
                  'h-4 w-4 text-muted-foreground transition-transform',
                  isPromptOpen && 'rotate-180'
                )}
              />
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2">
            <Textarea
              placeholder="e.g., Focus on action items, Write in bullet points, Extract key decisions..."
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              className="min-h-[80px] resize-none text-sm"
            />
          </CollapsibleContent>
        </Collapsible>

        {/* Generate Button */}
        <Button
          onClick={handleGenerate}
          disabled={!hasTranscript || isGenerating}
          className="w-full gap-2"
        >
          {isGenerating ? (
            <>
              <RefreshCw className="h-4 w-4 animate-spin" />
              Generating...
            </>
          ) : summary ? (
            <>
              <RefreshCw className="h-4 w-4" />
              Regenerate Summary
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              Generate Summary
            </>
          )}
        </Button>

        {/* Summary Output */}
        {isGenerating ? (
          <div className="space-y-3 pt-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/6" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </div>
        ) : summary ? (
          <div className="pt-2">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium">Result</span>
              <Button variant="ghost" size="sm" onClick={handleCopy}>
                <Copy className="h-3.5 w-3.5 mr-1.5" />
                Copy
              </Button>
            </div>
            <ScrollArea className="h-[200px]">
              <div className="rounded-lg bg-muted/50 p-4">
                <p className="text-sm whitespace-pre-wrap leading-relaxed">
                  {summary}
                </p>
              </div>
            </ScrollArea>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center">
            <Sparkles className="h-10 w-10 text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">
              {hasTranscript
                ? 'Click generate to create an AI summary'
                : 'Record or upload audio first'}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
