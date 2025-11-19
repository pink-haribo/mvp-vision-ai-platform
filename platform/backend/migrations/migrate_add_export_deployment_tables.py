"""
Add model export and deployment tables.

This migration creates:
1. export_jobs - Stores model export jobs (convert checkpoints to ONNX, TensorRT, etc.)
2. deployment_targets - Stores deployment configurations (download, platform endpoint, edge, container)
3. deployment_history - Stores deployment lifecycle events

Supports export formats:
- ONNX, TensorRT, CoreML, TFLite, TorchScript, OpenVINO

Supports deployment types:
- Download (self-hosted)
- Platform Endpoint (Triton Inference Server)
- Edge Package (iOS, Android)
- Container (Docker)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.core.config import settings


def migrate():
    """Run migration to add export and deployment tables."""
    print("\n" + "="*80)
    print("EXPORT & DEPLOYMENT TABLES MIGRATION")
    print("="*80)

    engine = create_engine(settings.DATABASE_URL)

    with engine.connect() as conn:
        print("\n[1/5] Creating export_jobs table...")

        # Create export_jobs table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS export_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_job_id INTEGER NOT NULL,

            -- Export configuration
            export_format TEXT NOT NULL,
            checkpoint_path TEXT NOT NULL,
            export_path TEXT,

            -- Version management
            version INTEGER NOT NULL DEFAULT 1,
            is_default INTEGER NOT NULL DEFAULT 0,

            -- Framework info (cached)
            framework TEXT NOT NULL,
            task_type TEXT NOT NULL,
            model_name TEXT NOT NULL,

            -- Configuration (JSON)
            export_config TEXT,
            optimization_config TEXT,
            validation_config TEXT,

            -- Status
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            process_id INTEGER,

            -- Results
            export_results TEXT,
            file_size_mb REAL,
            validation_passed INTEGER,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,

            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] export_jobs table created")

        print("\n[2/5] Creating deployment_targets table...")

        # Create deployment_targets table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS deployment_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            export_job_id INTEGER NOT NULL,
            training_job_id INTEGER NOT NULL,

            -- Deployment configuration
            deployment_type TEXT NOT NULL,
            deployment_name TEXT NOT NULL,
            deployment_config TEXT,

            -- Platform endpoint specific
            endpoint_url TEXT,
            api_key TEXT,

            -- Container specific
            container_image TEXT,
            container_registry TEXT,

            -- Edge package specific
            package_path TEXT,
            runtime_wrapper_language TEXT,

            -- Status
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,

            -- Usage tracking
            request_count INTEGER NOT NULL DEFAULT 0,
            total_inference_time_ms REAL NOT NULL DEFAULT 0.0,
            avg_latency_ms REAL,
            last_request_at TIMESTAMP,

            -- Resource usage
            cpu_limit TEXT,
            memory_limit TEXT,
            gpu_enabled INTEGER NOT NULL DEFAULT 0,

            -- Timestamps
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            deployed_at TIMESTAMP,
            deactivated_at TIMESTAMP,

            FOREIGN KEY (export_job_id) REFERENCES export_jobs(id) ON DELETE CASCADE,
            FOREIGN KEY (training_job_id) REFERENCES training_jobs(id) ON DELETE CASCADE
        )
        """))
        print("   [OK] deployment_targets table created")

        print("\n[3/5] Creating deployment_history table...")

        # Create deployment_history table
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS deployment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            deployment_id INTEGER NOT NULL,

            -- Event information
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,

            -- User tracking (nullable for system events)
            triggered_by INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (deployment_id) REFERENCES deployment_targets(id) ON DELETE CASCADE,
            FOREIGN KEY (triggered_by) REFERENCES users(id) ON DELETE SET NULL
        )
        """))
        print("   [OK] deployment_history table created")

        print("\n[4/5] Creating indexes...")

        # Export jobs indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_export_jobs_training_job
        ON export_jobs(training_job_id)
        """))
        print("   [OK] idx_export_jobs_training_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_export_jobs_format
        ON export_jobs(export_format)
        """))
        print("   [OK] idx_export_jobs_format")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_export_jobs_status
        ON export_jobs(status)
        """))
        print("   [OK] idx_export_jobs_status")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_export_jobs_version
        ON export_jobs(training_job_id, version)
        """))
        print("   [OK] idx_export_jobs_version")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_export_jobs_default
        ON export_jobs(is_default)
        """))
        print("   [OK] idx_export_jobs_default")

        # Deployment targets indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_targets_export_job
        ON deployment_targets(export_job_id)
        """))
        print("   [OK] idx_deployment_targets_export_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_targets_training_job
        ON deployment_targets(training_job_id)
        """))
        print("   [OK] idx_deployment_targets_training_job")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_targets_type
        ON deployment_targets(deployment_type)
        """))
        print("   [OK] idx_deployment_targets_type")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_targets_status
        ON deployment_targets(status)
        """))
        print("   [OK] idx_deployment_targets_status")

        # Deployment history indexes
        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_history_deployment
        ON deployment_history(deployment_id)
        """))
        print("   [OK] idx_deployment_history_deployment")

        conn.execute(text("""
        CREATE INDEX IF NOT EXISTS idx_deployment_history_event_type
        ON deployment_history(event_type)
        """))
        print("   [OK] idx_deployment_history_event_type")

        print("\n[5/5] Committing changes...")
        conn.commit()
        print("   [OK] Changes committed")

    print("\n" + "="*80)
    print("[SUCCESS] MIGRATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nExport & Deployment tables are now ready.")
    print("\nExport Tables:")
    print("  - export_jobs - Convert checkpoints to production formats")
    print("\nDeployment Tables:")
    print("  - deployment_targets - Manage deployments (download, endpoint, edge, container)")
    print("  - deployment_history - Track deployment lifecycle events")
    print("\nSupported export formats:")
    print("  - ONNX, TensorRT, CoreML, TFLite, TorchScript, OpenVINO")
    print("\nSupported deployment types:")
    print("  - Download (self-hosted)")
    print("  - Platform Endpoint (Triton Inference Server)")
    print("  - Edge Package (iOS, Android)")
    print("  - Container (Docker)")
    print("\n")


if __name__ == "__main__":
    migrate()
