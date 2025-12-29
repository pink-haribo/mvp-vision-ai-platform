# Temporal Workflows

**Workflow definitions** for orchestrating training jobs with Temporal.

## Structure

```
workflows/
├── training.py       # Main training workflow
├── preprocessing.py  # Data preparation
├── inference.py      # Inference workflow
├── monitoring.py     # Health checks
└── activities/       # Temporal activities
    ├── kubernetes.py # K8s operations
    ├── storage.py    # S3 operations
    └── database.py   # DB operations
```

## Training Workflow

```python
@workflow.defn
class TrainingWorkflow:
    @workflow.run
    async def run(self, job_id: int) -> TrainingResult:
        # 1. Validate dataset
        await workflow.execute_activity(
            validate_dataset,
            dataset_id,
            start_to_close_timeout=timedelta(minutes=5)
        )

        # 2. Create K8s Job
        k8s_job = await workflow.execute_activity(
            create_training_job,
            job_config,
            start_to_close_timeout=timedelta(minutes=2)
        )

        # 3. Monitor training (with heartbeat)
        result = await workflow.execute_activity(
            monitor_training,
            k8s_job.name,
            start_to_close_timeout=timedelta(hours=24),
            heartbeat_timeout=timedelta(minutes=5)
        )

        # 4. Cleanup
        await workflow.execute_activity(
            cleanup_resources,
            k8s_job.name
        )

        return result
```

## Features

- **Automatic Retries**: Exponential backoff on failures
- **Timeouts**: Activity and workflow level
- **Heartbeats**: Detect stuck activities
- **Cancellation**: Graceful workflow cancellation
- **History**: Full execution history for debugging

## Development

```bash
# Install Temporal
brew install temporal

# Start Temporal server
temporal server start-dev

# Run worker
cd platform/backend
poetry run python -m app.workflows.worker
```

## Workflow Versioning

When modifying workflows:
1. Create new workflow version
2. Deploy alongside old version
3. Let old workflows complete
4. Remove old version after migration
