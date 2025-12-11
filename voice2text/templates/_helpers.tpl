{{/*
Chart name (voice2text)
*/}}
{{- define "voice2text.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Full name (release-voice2text)
*/}}
{{- define "voice2text.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "voice2text.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Standard labels
*/}}
{{- define "voice2text.labels" -}}
app.kubernetes.io/name: {{ include "voice2text.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
