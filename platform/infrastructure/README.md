# Infrastructure as Code

**Deployment configurations** for multiple environments.

## Structure

```
infrastructure/
├── helm/              # Kubernetes Helm charts
│   ├── backend/       # Backend service chart
│   ├── frontend/      # Frontend service chart
│   ├── temporal/      # Temporal cluster
│   ├── trainers/      # Trainer job templates
│   └── observability/ # Grafana stack
├── terraform/         # Cloud infrastructure
│   ├── railway/       # Railway resources
│   ├── aws/          # AWS EKS + RDS + S3
│   └── onprem/       # On-premise K8s
└── k8s/              # Raw manifests (if needed)
```

## Helm Charts

### Backend Service
```bash
helm install backend ./helm/backend \
  --set image.tag=v1.0.0 \
  --set env.DATABASE_URL=$DATABASE_URL \
  --set env.STORAGE_ENDPOINT=$STORAGE_ENDPOINT
```

### Temporal Cluster
```bash
helm repo add temporalio https://go.temporal.io/helm-charts
helm install temporal temporalio/temporal \
  --set server.replicaCount=1 \
  --set cassandra.enabled=false \
  --set postgresql.enabled=true
```

### Observability Stack
```bash
helm install observability ./helm/observability \
  --set grafana.adminPassword=$GRAFANA_PASSWORD \
  --set loki.persistence.enabled=true
```

## Terraform

### Railway
```bash
cd terraform/railway
terraform init
terraform plan
terraform apply
```

### AWS
```bash
cd terraform/aws
terraform init
terraform plan -var-file=production.tfvars
terraform apply -var-file=production.tfvars
```

## Environment-Specific Configs

```
config/
├── dev.yaml          # Local development
├── railway.yaml      # Railway production
├── aws.yaml         # AWS production
└── onprem.yaml      # On-premise
```

Usage:
```bash
helm install backend ./helm/backend -f config/railway.yaml
```

## Secrets Management

**Never commit secrets!**

Use:
- Railway: Environment variables in dashboard
- AWS: AWS Secrets Manager + External Secrets Operator
- On-premise: Sealed Secrets or Vault
