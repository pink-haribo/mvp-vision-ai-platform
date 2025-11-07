# Cloud GPU Architecture

## Overview

Design for seamless execution of training jobs on both local and cloud GPU infrastructure with the same plugin-based model system.

**Key Requirements:**
- Support local and cloud GPU execution with same codebase
- Real-time monitoring regardless of execution location
- Efficient data transfer (datasets, checkpoints, results)
- Cost optimization (minimize idle GPU time)
- Fault tolerance and retry mechanisms
- Zero changes to model plugin interface

---

## Architecture Comparison

### Local Execution (Current)

```
Frontend
   ↓ HTTP
Backend API
   ↓ Python subprocess
Trainer (Local GPU)
   ↓ File System
Checkpoints/Results
   ↓ HTTP/WebSocket
Frontend (updates)
```

**Pros:**
- Simple architecture
- Low latency
- No network overhead
- Easy debugging

**Cons:**
- Limited by local GPU capacity
- No scalability
- GPU idle when not training

---

### Cloud GPU Execution (Target)

```
Frontend
   ↓ HTTP/WebSocket
Backend API
   ↓ Job Queue (Redis/Celery)
Job Orchestrator
   ↓ Cloud API (AWS/GCP/Azure)
GPU Cloud Instance (Container)
   ├─ Pull: Model code, Dataset
   ├─ Execute: Training
   └─ Push: Checkpoints, Logs, Metrics
   ↓ Object Storage (S3/GCS)
Backend API (monitors)
   ↓ WebSocket
Frontend (real-time updates)
```

**Pros:**
- Unlimited scalability
- Multiple parallel jobs
- GPU only when needed (cost-effective)
- Multiple GPU types available

**Cons:**
- More complex architecture
- Network latency for data transfer
- Cloud costs
- Debugging more difficult

---

## Hybrid Architecture Design

Support both local and cloud execution transparently:

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Training Config UI → Submit Job                       │ │
│  │  Real-time Monitoring ← WebSocket Updates              │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/WebSocket
┌───────────────────────────▼─────────────────────────────────┐
│                     Backend API Server                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ /training/create → Job Dispatcher                       ││
│  │   • Validate config                                     ││
│  │   • Select execution target (local/cloud)               ││
│  │   • Create job record in DB                             ││
│  │   • Enqueue job                                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Job Queue (Redis + Celery)                              ││
│  │   • Manage job lifecycle                                ││
│  │   • Retry failed jobs                                   ││
│  │   • Priority queue                                      ││
│  └─────────────────────────────────────────────────────────┘│
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────▼──────────┐           ┌───────────▼────────────┐
│  Local Executor    │           │  Cloud Orchestrator    │
│                    │           │                        │
│  • Check GPU avail │           │  • Launch GPU instance │
│  • Run subprocess  │           │  • Deploy container    │
│  • Monitor local   │           │  • Monitor remote      │
│  • Direct file I/O │           │  • Manage lifecycle    │
└─────────┬──────────┘           └───────────┬────────────┘
          │                                   │
┌─────────▼──────────┐           ┌───────────▼────────────┐
│  Local GPU         │           │  Cloud GPU Instance    │
│  ┌──────────────┐  │           │  ┌──────────────────┐  │
│  │ Training Job │  │           │  │ Docker Container │  │
│  │ (Model Plugin)  │           │  │ • Model Plugin   │  │
│  └──────────────┘  │           │  │ • Training Env   │  │
│                    │           │  │ • Agent (reports)│  │
│  Local Filesystem  │           │  └──────────────────┘  │
│  • Dataset         │           │                        │
│  • Checkpoints     │           │  Object Storage (S3)   │
│  • Logs            │           │  • Dataset (input)     │
└────────────────────┘           │  • Checkpoints (output)│
                                 │  • Logs (output)       │
                                 └────────────────────────┘
                                           │
                                 ┌─────────▼──────────┐
                                 │  Backend Monitors  │
                                 │  • Poll S3/logs    │
                                 │  • Update DB       │
                                 │  • Send WS updates │
                                 └────────────────────┘
```

---

## Component Design

### 1. Execution Strategy Pattern

```python
# training/executors/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class ExecutionStrategy(ABC):
    """Abstract executor for training jobs"""

    @abstractmethod
    async def can_execute(self) -> bool:
        """Check if this executor can run jobs"""
        pass

    @abstractmethod
    async def estimate_cost(self, job_config: Dict) -> float:
        """Estimate execution cost (time * rate)"""
        pass

    @abstractmethod
    async def submit_job(self, job_id: int, model_plugin: str, config: Dict) -> str:
        """
        Submit training job for execution
        Returns: execution_id (for tracking)
        """
        pass

    @abstractmethod
    async def get_status(self, execution_id: str) -> Dict:
        """Get current job status"""
        pass

    @abstractmethod
    async def cancel_job(self, execution_id: str):
        """Cancel running job"""
        pass

    @abstractmethod
    async def get_logs(self, execution_id: str, offset: int = 0) -> str:
        """Get training logs"""
        pass

    @abstractmethod
    async def get_metrics(self, execution_id: str) -> Dict:
        """Get current training metrics"""
        pass
```

---

### 2. Local Executor

```python
# training/executors/local_executor.py

import subprocess
import torch
from pathlib import Path

class LocalExecutor(ExecutionStrategy):
    """Execute training on local GPU"""

    async def can_execute(self) -> bool:
        """Check if local GPU is available"""
        return torch.cuda.is_available() and torch.cuda.device_count() > 0

    async def estimate_cost(self, job_config: Dict) -> float:
        """Local execution is free"""
        return 0.0

    async def submit_job(self, job_id: int, model_plugin: str, config: Dict) -> str:
        """
        Launch training as subprocess on local machine
        """
        # Prepare command
        cmd = [
            "python", "-m", "training.runner",
            "--job-id", str(job_id),
            "--model", model_plugin,
            "--config", json.dumps(config)
        ]

        # Launch subprocess (non-blocking)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
        )

        # Store process info
        execution_id = f"local-{job_id}-{process.pid}"
        self._active_jobs[execution_id] = {
            "process": process,
            "job_id": job_id,
            "started_at": datetime.utcnow()
        }

        return execution_id

    async def get_status(self, execution_id: str) -> Dict:
        """Check process status"""
        job_info = self._active_jobs.get(execution_id)
        if not job_info:
            return {"status": "not_found"}

        process = job_info["process"]
        if process.poll() is None:
            # Still running
            return {
                "status": "running",
                "started_at": job_info["started_at"]
            }
        else:
            # Completed
            return_code = process.returncode
            return {
                "status": "completed" if return_code == 0 else "failed",
                "return_code": return_code
            }

    async def cancel_job(self, execution_id: str):
        """Terminate process"""
        job_info = self._active_jobs.get(execution_id)
        if job_info:
            process = job_info["process"]
            process.terminate()
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()  # Force kill if not responding

    async def get_logs(self, execution_id: str, offset: int = 0) -> str:
        """Read logs from file"""
        job_info = self._active_jobs.get(execution_id)
        if not job_info:
            return ""

        log_file = Path(f"logs/training_{job_info['job_id']}.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                f.seek(offset)
                return f.read()
        return ""

    async def get_metrics(self, execution_id: str) -> Dict:
        """Read metrics from MLflow or file"""
        job_info = self._active_jobs.get(execution_id)
        if not job_info:
            return {}

        # Read from metrics file
        metrics_file = Path(f"outputs/training_{job_info['job_id']}_metrics.json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}
```

---

### 3. Cloud Executor (AWS Example)

```python
# training/executors/aws_executor.py

import boto3
import json
from typing import Dict, Any

class AWSExecutor(ExecutionStrategy):
    """Execute training on AWS GPU instances"""

    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.s3 = boto3.client('s3')
        self.ecs = boto3.client('ecs')  # ECS for container orchestration
        self.bucket_name = "vision-platform-training"

    async def can_execute(self) -> bool:
        """Always available (assuming AWS account configured)"""
        return True

    async def estimate_cost(self, job_config: Dict) -> float:
        """
        Estimate AWS cost based on instance type and duration
        """
        instance_type = job_config.get('instance_type', 'g4dn.xlarge')
        epochs = job_config.get('epochs', 100)
        estimated_hours = self._estimate_duration_hours(job_config)

        # Pricing (example - actual prices vary)
        pricing = {
            'g4dn.xlarge': 0.526,    # $0.526/hour (1 GPU)
            'g4dn.2xlarge': 0.752,   # $0.752/hour (1 GPU, more CPU/RAM)
            'p3.2xlarge': 3.06,      # $3.06/hour (1 V100 GPU)
            'p3.8xlarge': 12.24,     # $12.24/hour (4 V100 GPUs)
        }

        hourly_rate = pricing.get(instance_type, 1.0)
        return estimated_hours * hourly_rate

    async def submit_job(self, job_id: int, model_plugin: str, config: Dict) -> str:
        """
        Deploy training job to AWS

        Steps:
        1. Package model code and config
        2. Upload dataset to S3 (if not already there)
        3. Launch ECS task with GPU
        4. Return task ARN for tracking
        """

        # 1. Prepare job package
        job_package = await self._prepare_job_package(
            job_id, model_plugin, config
        )

        # 2. Upload to S3
        s3_key = f"jobs/{job_id}/package.tar.gz"
        self.s3.upload_file(
            job_package,
            self.bucket_name,
            s3_key
        )

        # 3. Ensure dataset is in S3
        dataset_s3_key = await self._ensure_dataset_in_s3(
            config['dataset_path']
        )

        # 4. Launch ECS task
        task_definition = self._get_task_definition(config)

        response = self.ecs.run_task(
            cluster='vision-platform-training',
            taskDefinition=task_definition,
            launchType='EC2',  # or 'FARGATE' for serverless
            overrides={
                'containerOverrides': [{
                    'name': 'trainer',
                    'environment': [
                        {'name': 'JOB_ID', 'value': str(job_id)},
                        {'name': 'MODEL_PLUGIN', 'value': model_plugin},
                        {'name': 'S3_BUCKET', 'value': self.bucket_name},
                        {'name': 'JOB_PACKAGE_KEY', 'value': s3_key},
                        {'name': 'DATASET_KEY', 'value': dataset_s3_key},
                        {'name': 'CONFIG', 'value': json.dumps(config)},
                    ]
                }]
            },
            count=1,
            enableECSManagedTags=True,
            tags=[
                {'key': 'JobId', 'value': str(job_id)},
                {'key': 'Model', 'value': model_plugin}
            ]
        )

        task_arn = response['tasks'][0]['taskArn']
        execution_id = f"aws-ecs-{task_arn.split('/')[-1]}"

        # Store mapping
        await self._store_execution_mapping(job_id, execution_id, task_arn)

        return execution_id

    async def get_status(self, execution_id: str) -> Dict:
        """Query ECS task status"""
        task_arn = await self._get_task_arn(execution_id)

        response = self.ecs.describe_tasks(
            cluster='vision-platform-training',
            tasks=[task_arn]
        )

        if not response['tasks']:
            return {"status": "not_found"}

        task = response['tasks'][0]
        status_map = {
            'PENDING': 'pending',
            'RUNNING': 'running',
            'STOPPED': 'completed'  # Check exit code for success/failure
        }

        ecs_status = task['lastStatus']
        status = status_map.get(ecs_status, 'unknown')

        result = {
            "status": status,
            "started_at": task.get('startedAt'),
            "stopped_at": task.get('stoppedAt'),
        }

        # If stopped, check exit code
        if status == 'completed' and task.get('containers'):
            exit_code = task['containers'][0].get('exitCode', 0)
            result['status'] = 'completed' if exit_code == 0 else 'failed'
            result['exit_code'] = exit_code

        return result

    async def cancel_job(self, execution_id: str):
        """Stop ECS task"""
        task_arn = await self._get_task_arn(execution_id)

        self.ecs.stop_task(
            cluster='vision-platform-training',
            task=task_arn,
            reason='User requested cancellation'
        )

    async def get_logs(self, execution_id: str, offset: int = 0) -> str:
        """
        Read logs from CloudWatch Logs
        """
        job_id = await self._get_job_id(execution_id)
        log_group = '/ecs/vision-platform-training'
        log_stream = f'trainer/{job_id}'

        logs_client = boto3.client('logs')

        try:
            response = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                startFromHead=True,
                limit=1000
            )

            events = response.get('events', [])
            logs = '\n'.join([e['message'] for e in events])
            return logs[offset:]

        except logs_client.exceptions.ResourceNotFoundException:
            return ""

    async def get_metrics(self, execution_id: str) -> Dict:
        """
        Read metrics from S3 (uploaded by training container)
        """
        job_id = await self._get_job_id(execution_id)
        metrics_key = f"jobs/{job_id}/metrics.json"

        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=metrics_key
            )
            metrics = json.loads(response['Body'].read())
            return metrics

        except self.s3.exceptions.NoSuchKey:
            return {}

    # Helper methods
    async def _prepare_job_package(self, job_id, model_plugin, config):
        """Package model code, dependencies, config into tarball"""
        # Implementation details...
        pass

    async def _ensure_dataset_in_s3(self, local_path):
        """Upload dataset to S3 if not already there"""
        # Check if already uploaded, if not upload
        # Return S3 key
        pass

    def _get_task_definition(self, config):
        """Get ECS task definition based on GPU requirements"""
        instance_type = config.get('instance_type', 'g4dn.xlarge')

        # Map to task definition
        task_defs = {
            'g4dn.xlarge': 'vision-trainer-g4dn',
            'p3.2xlarge': 'vision-trainer-p3',
        }
        return task_defs.get(instance_type, 'vision-trainer-default')
```

---

### 4. Training Container (Cloud)

Docker image that runs in cloud GPU instances:

```dockerfile
# training/docker/Dockerfile

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip

# Install PyTorch and dependencies
RUN pip install torch torchvision timm ultralytics mlflow boto3

# Copy platform code
COPY training/ /app/training/
COPY app/ /app/app/

WORKDIR /app

# Entry point script
COPY training/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

```bash
# training/docker/entrypoint.sh

#!/bin/bash
set -e

# Environment variables (set by ECS):
# - JOB_ID
# - MODEL_PLUGIN
# - S3_BUCKET
# - JOB_PACKAGE_KEY
# - DATASET_KEY
# - CONFIG

echo "Starting training job ${JOB_ID}"

# 1. Download job package from S3
aws s3 cp s3://${S3_BUCKET}/${JOB_PACKAGE_KEY} /tmp/package.tar.gz
tar -xzf /tmp/package.tar.gz -C /app

# 2. Download dataset from S3
mkdir -p /data
aws s3 sync s3://${S3_BUCKET}/${DATASET_KEY} /data/dataset

# 3. Run training with cloud agent
python -m training.cloud_runner \
    --job-id ${JOB_ID} \
    --model ${MODEL_PLUGIN} \
    --config "${CONFIG}" \
    --dataset /data/dataset \
    --output /data/output \
    --s3-bucket ${S3_BUCKET} \
    --s3-prefix jobs/${JOB_ID}

# 4. Upload results to S3
echo "Uploading results to S3..."
aws s3 sync /data/output s3://${S3_BUCKET}/jobs/${JOB_ID}/output

echo "Training job ${JOB_ID} completed"
```

```python
# training/cloud_runner.py

"""
Training runner for cloud GPU instances
Includes agent that reports metrics back to platform
"""

import argparse
import json
import boto3
import time
from pathlib import Path
from training.registry import registry

class CloudTrainingAgent:
    """Agent that runs in cloud GPU instance"""

    def __init__(self, job_id, s3_bucket, s3_prefix):
        self.job_id = job_id
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket
        self.prefix = s3_prefix

    def report_metrics(self, epoch, metrics):
        """Upload metrics to S3 for platform to read"""
        metrics_data = {
            'job_id': self.job_id,
            'epoch': epoch,
            'timestamp': time.time(),
            'metrics': metrics
        }

        key = f"{self.prefix}/metrics.json"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metrics_data),
            ContentType='application/json'
        )

    def upload_checkpoint(self, epoch, checkpoint_path):
        """Upload checkpoint to S3"""
        key = f"{self.prefix}/checkpoints/epoch_{epoch}.pth"
        self.s3.upload_file(
            checkpoint_path,
            self.bucket,
            key
        )

    def report_logs(self, logs):
        """Upload logs to S3"""
        key = f"{self.prefix}/logs.txt"
        # Append to existing logs
        existing_logs = ""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            existing_logs = obj['Body'].read().decode('utf-8')
        except:
            pass

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=existing_logs + logs
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-id', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--s3-bucket', type=str, required=True)
    parser.add_argument('--s3-prefix', type=str, required=True)
    args = parser.parse_args()

    # Parse config
    config = json.loads(args.config)

    # Initialize agent
    agent = CloudTrainingAgent(args.job_id, args.s3_bucket, args.s3_prefix)

    # Load model from registry
    model_cls = registry.get_model(args.model)
    model = model_cls(config)

    # Prepare data loaders
    train_loader, val_loader = prepare_dataloaders(args.dataset, config)

    # Training callbacks for cloud reporting
    class CloudCallbacks:
        def on_epoch_end(self, epoch, metrics):
            print(f"Epoch {epoch}: {metrics}")
            agent.report_metrics(epoch, metrics)

        def on_checkpoint_save(self, epoch, path):
            print(f"Uploading checkpoint for epoch {epoch}")
            agent.upload_checkpoint(epoch, path)

    callbacks = CloudCallbacks()

    # Run training
    print(f"Starting training for job {args.job_id}")
    try:
        model.train(train_loader, val_loader, callbacks=callbacks)
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed: {e}")
        agent.report_logs(f"ERROR: {str(e)}")
        raise

if __name__ == '__main__':
    main()
```

---

### 5. Job Dispatcher (Selects Executor)

```python
# app/api/training.py (updated)

from training.executors import LocalExecutor, AWSExecutor, GCPExecutor

class JobDispatcher:
    """Decides where to execute training jobs"""

    def __init__(self):
        self.executors = {
            'local': LocalExecutor(),
            'aws': AWSExecutor(),
            'gcp': GCPExecutor(),
        }

    async def select_executor(self, config: Dict) -> ExecutionStrategy:
        """
        Select best executor based on:
        - User preference
        - GPU availability
        - Cost constraints
        - Job priority
        """

        # User explicitly requested location
        if 'executor' in config:
            return self.executors[config['executor']]

        # Auto-select based on availability and cost

        # 1. Try local first (free)
        local_executor = self.executors['local']
        if await local_executor.can_execute():
            return local_executor

        # 2. No local GPU available, use cloud
        # Select cheapest option that meets requirements
        cloud_options = []
        for name, executor in self.executors.items():
            if name != 'local':
                cost = await executor.estimate_cost(config)
                cloud_options.append((name, executor, cost))

        # Sort by cost
        cloud_options.sort(key=lambda x: x[2])

        # Return cheapest
        if cloud_options:
            return cloud_options[0][1]

        raise RuntimeError("No available executors")

# Updated training endpoint
@router.post("/training")
async def create_training_job(
    job_config: TrainingJobCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create and submit training job"""

    # Create job record
    job = TrainingJob(
        user_id=current_user.id,
        model_name=job_config.model_name,
        config=job_config.config,
        status='pending'
    )
    db.add(job)
    db.commit()

    # Select executor
    dispatcher = JobDispatcher()
    executor = await dispatcher.select_executor(job_config.config)

    # Submit job
    execution_id = await executor.submit_job(
        job.id,
        job_config.model_name,
        job_config.config
    )

    # Update job record
    job.execution_id = execution_id
    job.executor_type = executor.__class__.__name__
    job.status = 'queued'
    db.commit()

    return {"job_id": job.id, "execution_id": execution_id}
```

---

### 6. Monitoring Service

```python
# app/services/training_monitor.py

import asyncio
from typing import Dict
from app.db.database import get_db
from training.executors import get_executor_by_type

class TrainingMonitor:
    """
    Background service that monitors active training jobs
    Updates database and sends WebSocket notifications
    """

    def __init__(self):
        self.running = False

    async def start(self):
        """Start monitoring loop"""
        self.running = True
        while self.running:
            await self.monitor_jobs()
            await asyncio.sleep(10)  # Check every 10 seconds

    async def monitor_jobs(self):
        """Check status of all active jobs"""
        db = next(get_db())

        # Get all running/queued jobs
        active_jobs = db.query(TrainingJob).filter(
            TrainingJob.status.in_(['queued', 'running'])
        ).all()

        for job in active_jobs:
            try:
                # Get executor
                executor = get_executor_by_type(job.executor_type)

                # Check status
                status = await executor.get_status(job.execution_id)

                # Update job
                if status['status'] != job.status:
                    job.status = status['status']
                    db.commit()

                    # Send WebSocket notification
                    await self.notify_status_change(job.id, status)

                # Get latest metrics
                if job.status == 'running':
                    metrics = await executor.get_metrics(job.execution_id)
                    if metrics:
                        await self.update_metrics(job.id, metrics)
                        await self.notify_metrics_update(job.id, metrics)

            except Exception as e:
                print(f"Error monitoring job {job.id}: {e}")

    async def notify_status_change(self, job_id, status):
        """Send WebSocket notification about status change"""
        # Implementation using WebSocket manager
        pass

    async def notify_metrics_update(self, job_id, metrics):
        """Send WebSocket notification with latest metrics"""
        # Implementation using WebSocket manager
        pass

    async def update_metrics(self, job_id, metrics):
        """Store metrics in database"""
        # Implementation
        pass
```

---

## Data Flow

### Training Submission (Cloud)

```
1. User configures training in Frontend
   ↓
2. Frontend → POST /api/v1/training
   ↓
3. Backend API:
   - Validates config
   - Creates TrainingJob record
   - Selects Executor (e.g., AWS)
   ↓
4. AWS Executor:
   - Packages model code
   - Uploads to S3: jobs/{job_id}/package.tar.gz
   - Uploads dataset to S3 (if needed): datasets/{dataset_id}/
   - Launches ECS task with GPU
   ↓
5. ECS Task starts:
   - Downloads package from S3
   - Downloads dataset from S3
   - Initializes model from plugin
   - Starts training
   ↓
6. During Training:
   - CloudTrainingAgent reports metrics → S3
   - CloudTrainingAgent uploads checkpoints → S3
   - CloudTrainingAgent streams logs → CloudWatch
   ↓
7. Monitoring Service:
   - Polls S3 for new metrics every 10s
   - Updates TrainingJob in database
   - Sends WebSocket updates to Frontend
   ↓
8. Frontend:
   - Receives WebSocket updates
   - Updates training dashboard in real-time
   ↓
9. Training Completes:
   - Final checkpoint → S3
   - Final metrics → S3
   - ECS task stops
   - Backend updates job status to 'completed'
   - Frontend shows completion notification
```

### Real-time Updates Flow

```
Cloud GPU Instance
   ├─ Epoch 1 complete → Upload metrics.json to S3
   ├─ Epoch 5 complete → Upload checkpoint_epoch5.pth to S3
   └─ Epoch 10 complete → Upload metrics.json to S3

Backend Monitor (polling S3 every 10s)
   ├─ Detects new metrics.json
   ├─ Updates TrainingJob.current_epoch
   ├─ Updates TrainingJob.current_metrics
   └─ Sends WebSocket message

WebSocket Manager
   └─ Broadcasts to connected clients for this job

Frontend (subscribed to job updates)
   ├─ Receives WebSocket message
   ├─ Updates training progress chart
   └─ Updates metrics display
```

---

## Plugin Compatibility

**Key Point:** Model plugins don't change at all!

The plugin interface remains the same whether running locally or in cloud:

```python
class MyModel(BaseModel):
    def train(self, train_loader, val_loader, callbacks):
        for epoch in range(self.config['epochs']):
            # Training logic
            loss = ...

            # Callbacks work the same locally or in cloud
            if callbacks:
                callbacks.on_epoch_end(epoch, {'loss': loss})
```

**Locally:** Callbacks update local database + WebSocket
**Cloud:** Callbacks upload to S3, monitor service polls and updates

---

## Cost Optimization Strategies

### 1. Dataset Caching

```python
# Avoid re-uploading datasets
class DatasetCache:
    def get_or_upload(self, local_path):
        # Check if already in S3
        dataset_hash = hash_directory(local_path)
        s3_key = f"datasets/{dataset_hash}/"

        if self.exists_in_s3(s3_key):
            return s3_key  # Reuse existing

        # Upload new
        self.upload_to_s3(local_path, s3_key)
        return s3_key
```

### 2. Spot Instances

```python
# AWS Executor with spot instance support
class AWSExecutor:
    async def submit_job(self, job_id, model_plugin, config):
        # Use spot instances for non-critical jobs (90% cost reduction)
        use_spot = config.get('allow_spot_instances', True)

        if use_spot:
            launch_type = 'FARGATE_SPOT'
        else:
            launch_type = 'FARGATE'

        # Handle spot interruptions with checkpoint resume
```

### 3. Auto-shutdown

```python
# Container automatically shuts down after training
# No idle time billing
```

### 4. Instance Right-sizing

```python
# Auto-select instance based on model size
def recommend_instance_type(model_metadata, config):
    gpu_memory_gb = model_metadata.min_gpu_memory_gb or 8
    batch_size = config.get('batch_size', 32)

    # Map to appropriate instance
    if gpu_memory_gb <= 8 and batch_size <= 32:
        return 'g4dn.xlarge'  # $0.526/hr
    elif gpu_memory_gb <= 16:
        return 'g4dn.2xlarge'  # $0.752/hr
    else:
        return 'p3.2xlarge'  # $3.06/hr (V100)
```

---

## Failure Handling

### Retry Strategy

```python
class CloudExecutor:
    async def submit_job(self, job_id, model_plugin, config):
        max_retries = config.get('max_retries', 3)
        retry_count = 0

        while retry_count < max_retries:
            try:
                execution_id = await self._launch_task(...)
                return execution_id
            except SpotInterruption:
                # Spot instance was interrupted, retry
                retry_count += 1
                await asyncio.sleep(60)  # Wait before retry
            except Exception as e:
                # Other failures
                if retry_count < max_retries - 1:
                    retry_count += 1
                    await asyncio.sleep(60)
                else:
                    raise
```

### Checkpoint Resume

```python
# If job fails, can resume from last checkpoint
@router.post("/training/{job_id}/resume")
async def resume_training(job_id: int):
    job = get_job(job_id)

    # Find latest checkpoint in S3
    latest_checkpoint = find_latest_checkpoint(job_id)

    # Resubmit with resume flag
    config = job.config.copy()
    config['resume_from'] = latest_checkpoint

    execution_id = await executor.submit_job(job_id, job.model_name, config)
```

---

## Frontend Integration

No changes needed in Frontend! It uses the same API and WebSocket regardless of execution location.

```typescript
// Frontend code - works with both local and cloud
const startTraining = async (config) => {
  // Submit job
  const response = await fetch('/api/v1/training', {
    method: 'POST',
    body: JSON.stringify(config)
  });

  const { job_id } = await response.json();

  // Subscribe to updates (same WebSocket for local or cloud)
  const ws = new WebSocket(`ws://api/v1/training/${job_id}/stream`);
  ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    updateTrainingDashboard(update);
  };
};
```

---

## Summary

### Key Benefits

✅ **Transparent Execution**: Same plugin interface, works everywhere
✅ **Automatic Selection**: Platform picks best execution location
✅ **Real-time Monitoring**: WebSocket updates regardless of location
✅ **Cost Effective**: GPU only when training, spot instances, auto-shutdown
✅ **Scalable**: Run multiple jobs in parallel on cloud
✅ **Fault Tolerant**: Automatic retries, checkpoint resume
✅ **Zero Frontend Changes**: Same UI works for local and cloud

### Implementation Priority

**Phase 0 (Foundation):**
- ExecutionStrategy interface
- LocalExecutor (already mostly done)
- Job Dispatcher

**Phase 1 (Basic Cloud):**
- AWSExecutor with ECS
- CloudTrainingAgent
- S3 integration for checkpoints
- Monitoring service

**Phase 2 (Optimization):**
- Spot instance support
- Dataset caching
- Auto-scaling
- Cost analytics

**Phase 3 (Multi-cloud):**
- GCPExecutor
- AzureExecutor
- Hybrid strategies

---

*Document Version: 1.0*
*Last Updated: 2025-10-24*
*Author: Development Team*
