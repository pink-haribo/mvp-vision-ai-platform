# Observability

**Monitoring, logging, and tracing** configurations.

## Structure

```
observability/
├── grafana/           # Dashboards and alerts
│   ├── dashboards/   # JSON dashboard definitions
│   └── alerts/       # Alert rules
├── prometheus/        # Metrics collection
│   └── rules/        # Recording and alerting rules
├── loki/             # Log aggregation
│   └── config.yaml   # Loki configuration
└── otel/             # OpenTelemetry
    └── collector.yaml # OTel Collector config
```

## Components

### Grafana Dashboards

**Training Overview**
- Active jobs count
- Success/failure rate
- Average training time
- Resource utilization

**System Health**
- Backend API latency
- Database connection pool
- Storage throughput
- Worker queue depth

**Training Job Detail**
- Epoch progression
- Loss curves
- GPU utilization
- Memory usage

### Prometheus Metrics

```promql
# Job success rate (last 24h)
rate(training_jobs_completed_total{status="succeeded"}[24h])
/ rate(training_jobs_completed_total[24h])

# Average training duration
histogram_quantile(0.95,
  rate(training_duration_seconds_bucket[1h]))

# Active training jobs
sum(training_jobs_active)
```

### Loki Queries

```logql
# All logs for job
{job="training", job_id="123"}

# Errors only
{job="training"} |= "ERROR"

# Training progress
{job="training"} |= "Epoch"
```

### OpenTelemetry

Traces workflow:
```
Frontend → Backend → Temporal → K8s Job
  [trace_id: abc-def-ghi propagated through all]
```

## Setup

### Local (Docker Compose)
```bash
cd infrastructure/helm/observability
helm install grafana grafana/grafana
helm install loki grafana/loki-stack
helm install prometheus prometheus-community/prometheus
```

### Grafana Cloud
```bash
# Set environment variables
export GRAFANA_CLOUD_API_KEY=...
export LOKI_CLOUD_URL=...
export PROMETHEUS_CLOUD_URL=...

# OTel Collector auto-sends data
```

### Self-Hosted (K8s)
```bash
helm install observability ./helm/observability
```

## Access

**Local**:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Loki: http://localhost:3100

**Production**:
- Grafana: https://grafana.example.com
- Use SSO or OAuth

## Alerts

### Critical
- Training job failures > 10% in 1h
- Backend API errors > 1% in 5m
- Database connection pool exhausted
- Storage quota > 90%

### Warning
- Training duration > 2x average
- Worker queue depth > 100
- Memory usage > 80%

## Retention

- **Metrics**: 30 days
- **Logs**: 7 days (structured), 3 days (unstructured)
- **Traces**: 24 hours
