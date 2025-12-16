import React from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Mic, Shield, Lock, Zap } from 'lucide-react';

const Login: React.FC = () => {
  const { login, isLoading } = useAuth();

  const features = [
    {
      icon: Mic,
      title: 'Record & Transcribe',
      description: 'Capture audio or video and get accurate transcriptions instantly.',
    },
    {
      icon: Zap,
      title: 'AI-Powered Summaries',
      description: 'Generate intelligent summaries with custom prompts.',
    },
    {
      icon: Shield,
      title: 'Enterprise Security',
      description: 'Your data stays on-premise with enterprise-grade authentication.',
    },
  ];

  return (
    <div className="min-h-screen bg-background flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-primary relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary via-primary to-accent/30" />
        <div className="relative z-10 flex flex-col justify-center p-12 text-primary-foreground">
          <div className="flex items-center gap-3 mb-8">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary-foreground/10 backdrop-blur">
              <Mic className="h-7 w-7" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">VoiceScribe</h1>
              <p className="text-primary-foreground/70 text-sm">
                Enterprise Transcription Platform
              </p>
            </div>
          </div>

          <h2 className="text-4xl font-bold leading-tight mb-6">
            Transform voice into
            <br />
            actionable insights
          </h2>
          <p className="text-lg text-primary-foreground/80 mb-12 max-w-md">
            Record, transcribe, and summarize meetings, interviews, and voice notes with
            enterprise-grade security.
          </p>

          <div className="space-y-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="flex items-start gap-4 animate-fade-in"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-primary-foreground/10">
                  <feature.icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-semibold mb-1">{feature.title}</h3>
                  <p className="text-sm text-primary-foreground/70">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Decorative elements */}
        <div className="absolute -bottom-32 -right-32 h-64 w-64 rounded-full bg-accent/20 blur-3xl" />
        <div className="absolute -top-32 -left-32 h-64 w-64 rounded-full bg-primary-foreground/5 blur-3xl" />
      </div>

      {/* Right Panel - Login */}
      <div className="flex w-full lg:w-1/2 items-center justify-center p-8">
        <Card className="w-full max-w-md border-0 shadow-none lg:shadow-lg lg:border">
          <CardHeader className="text-center space-y-4">
            <div className="lg:hidden flex items-center justify-center gap-3 mb-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary">
                <Mic className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-xl font-bold">VoiceScribe</span>
            </div>
            <CardTitle className="text-2xl">Welcome back</CardTitle>
            <CardDescription>
              Sign in with your enterprise credentials to continue
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <Button
              onClick={login}
              disabled={isLoading}
              className="w-full h-12 text-base gap-3"
            >
              <Lock className="h-5 w-5" />
              Sign in with SSO
            </Button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-2 text-muted-foreground">
                  Secured by Keycloak
                </span>
              </div>
            </div>

            <p className="text-center text-xs text-muted-foreground">
              By signing in, you agree to your organization's terms of service and privacy
              policy.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Login;
