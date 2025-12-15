{{/*
===============================================================================
Common Helpers
===============================================================================
*/}}

{{/*
Expand the name of the chart.
*/}}
{{- define "vision-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "vision-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "vision-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "vision-ai.labels" -}}
helm.sh/chart: {{ include "vision-ai.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
===============================================================================
Backend Helpers
===============================================================================
*/}}

{{/*
Backend fullname
*/}}
{{- define "vision-ai.backend.fullname" -}}
{{- printf "%s-backend" (include "vision-ai.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Backend labels
*/}}
{{- define "vision-ai.backend.labels" -}}
{{ include "vision-ai.labels" . }}
{{ include "vision-ai.backend.selectorLabels" . }}
{{- end }}

{{/*
Backend selector labels
*/}}
{{- define "vision-ai.backend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vision-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Backend service account name
*/}}
{{- define "vision-ai.backend.serviceAccountName" -}}
{{- if .Values.backend.serviceAccount.create }}
{{- default (include "vision-ai.backend.fullname" .) .Values.backend.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.backend.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
===============================================================================
Frontend Helpers
===============================================================================
*/}}

{{/*
Frontend fullname
*/}}
{{- define "vision-ai.frontend.fullname" -}}
{{- printf "%s-frontend" (include "vision-ai.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Frontend labels
*/}}
{{- define "vision-ai.frontend.labels" -}}
{{ include "vision-ai.labels" . }}
{{ include "vision-ai.frontend.selectorLabels" . }}
{{- end }}

{{/*
Frontend selector labels
*/}}
{{- define "vision-ai.frontend.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vision-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Frontend service account name
*/}}
{{- define "vision-ai.frontend.serviceAccountName" -}}
{{- if .Values.frontend.serviceAccount.create }}
{{- default (include "vision-ai.frontend.fullname" .) .Values.frontend.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.frontend.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
===============================================================================
Training Infrastructure Helpers
===============================================================================
*/}}

{{/*
Training fullname
*/}}
{{- define "vision-ai.training.fullname" -}}
{{- printf "%s-training" (include "vision-ai.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Training labels
*/}}
{{- define "vision-ai.training.labels" -}}
{{ include "vision-ai.labels" . }}
{{ include "vision-ai.training.selectorLabels" . }}
{{- end }}

{{/*
Training selector labels
*/}}
{{- define "vision-ai.training.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vision-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: training
{{- end }}

{{/*
Training namespace
*/}}
{{- define "vision-ai.training.namespace" -}}
{{- default .Release.Namespace .Values.training.namespace }}
{{- end }}

{{/*
Training service account name
*/}}
{{- define "vision-ai.training.serviceAccountName" -}}
{{- if .Values.training.serviceAccount.create }}
{{- default (printf "%s-job-sa" (include "vision-ai.training.fullname" .)) .Values.training.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.training.serviceAccount.name }}
{{- end }}
{{- end }}
