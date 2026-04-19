{{/*
Expand the name of the chart.
*/}}
{{- define "h2ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "h2ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Chart label
*/}}
{{- define "h2ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "h2ai.labels" -}}
helm.sh/chart: {{ include "h2ai.chart" . }}
{{ include "h2ai.selectorLabels" . }}
app.kubernetes.io/version: {{ .Values.image.tag | default .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "h2ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "h2ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Image reference
*/}}
{{- define "h2ai.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end }}

{{/*
NATS URL — use dependency or bring-your-own
*/}}
{{- define "h2ai.natsUrl" -}}
{{- if .Values.nats.enabled }}
{{- printf "nats://%s-nats.%s.svc.cluster.local:4222" .Release.Name .Release.Namespace }}
{{- else }}
{{- required "nats.natsUrl is required when nats.enabled=false" .Values.nats.natsUrl }}
{{- end }}
{{- end }}
