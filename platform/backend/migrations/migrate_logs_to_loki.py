"""
Migrate training logs from SQLite database to Loki.

This script reads logs from the database and pushes them to Loki
for real-time log aggregation and querying.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from app.db.database import SessionLocal
from app.db import models


def migrate_job_logs_to_loki(job_id: int, loki_url: str = "http://localhost:3100"):
    """
    Migrate logs for a specific job from DB to Loki.

    Args:
        job_id: Training job ID
        loki_url: Loki push API URL
    """
    db = SessionLocal()
    try:
        # Fetch all logs for this job
        logs = db.query(models.TrainingLog).filter(
            models.TrainingLog.job_id == job_id
        ).order_by(models.TrainingLog.created_at).all()

        if not logs:
            print(f"No logs found for job_id={job_id}")
            return

        print(f"Found {len(logs)} logs for job_id={job_id}")

        # Group logs by type for efficient batching
        stdout_logs = []
        stderr_logs = []

        for log in logs:
            log_data = (log.created_at, log.content)
            if log.log_type == 'stdout':
                stdout_logs.append(log_data)
            else:
                stderr_logs.append(log_data)

        # Push stdout logs
        if stdout_logs:
            print(f"Pushing {len(stdout_logs)} stdout logs to Loki...")
            push_to_loki(job_id, stdout_logs, 'stdout', loki_url)

        # Push stderr logs
        if stderr_logs:
            print(f"Pushing {len(stderr_logs)} stderr logs to Loki...")
            push_to_loki(job_id, stderr_logs, 'stderr', loki_url)

        print(f"[OK] Successfully migrated logs for job_id={job_id}")

    except Exception as e:
        print(f"[ERROR] Error migrating logs: {e}")
        raise
    finally:
        db.close()


def push_to_loki(job_id: int, logs: list, log_type: str, loki_url: str):
    """
    Push logs to Loki using the Push API.

    Args:
        job_id: Training job ID
        logs: List of (timestamp, content) tuples
        log_type: 'stdout' or 'stderr'
        loki_url: Loki base URL
    """
    url = f"{loki_url}/loki/api/v1/push"

    # Build values array with original timestamps
    values = []
    for created_at, content in logs:
        # Convert datetime to nanoseconds
        if isinstance(created_at, datetime):
            timestamp_ns = str(int(created_at.timestamp() * 1_000_000_000))
        else:
            timestamp_ns = str(int(datetime.utcnow().timestamp() * 1_000_000_000))

        values.append([timestamp_ns, content])

    # Single stream for efficiency
    stream = {
        "stream": {
            "job": "training",
            "job_id": str(job_id),
            "log_type": log_type,
            "source": "backend"
        },
        "values": values
    }

    payload = {"streams": [stream]}

    # Send to Loki
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"  [OK] Pushed {len(values)} {log_type} logs (status: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Failed to push {log_type} logs: {e}")
        raise


def main():
    """Main entry point."""
    # Check if Loki is running
    loki_url = os.getenv('LOKI_URL', 'http://localhost:3100')

    try:
        response = requests.get(f"{loki_url}/ready", timeout=5)
        response_text = response.text.strip()
        print(f"Loki readiness check: '{response_text}' (status: {response.status_code})")
        if response_text != 'ready':
            print(f"[WARN] Loki might not be ready, but continuing anyway...")
        else:
            print(f"[OK] Loki is ready at {loki_url}")
    except Exception as e:
        print(f"[WARN] Cannot verify Loki status at {loki_url}: {e}")
        print("Continuing anyway...")

    # Migrate logs for job_id=16
    job_id = 16
    print(f"\nMigrating logs for job_id={job_id}...")
    migrate_job_logs_to_loki(job_id, loki_url)

    # Verify logs in Loki
    print(f"\nVerifying logs in Loki...")
    query_url = f"{loki_url}/loki/api/v1/query_range"
    params = {
        "query": f'{{job="training", job_id="{job_id}"}}',
        "limit": 10,
    }

    try:
        response = requests.get(query_url, params=params, timeout=10)
        data = response.json()

        if data.get('status') == 'success':
            results = data.get('data', {}).get('result', [])
            total_entries = sum(len(r.get('values', [])) for r in results)
            print(f"[OK] Loki query returned {len(results)} streams with {total_entries} total entries")

            if total_entries > 0:
                print(f"\nSample logs:")
                for stream in results[:1]:  # Show first stream
                    for entry in stream.get('values', [])[:3]:  # Show first 3 entries
                        timestamp, log = entry
                        dt = datetime.fromtimestamp(int(timestamp) / 1e9)
                        print(f"  [{dt.isoformat()}] {log[:100]}")
        else:
            print(f"[ERROR] Loki query failed: {data}")
    except Exception as e:
        print(f"[ERROR] Failed to verify logs: {e}")


if __name__ == "__main__":
    main()
