"""
Kubernetes Training Manager

Phase 12: Temporal Orchestration & Backend Modernization

Manages training jobs by creating Kubernetes Jobs (Tier 1+).
This implementation is used when TRAINING_MODE=kubernetes.

Architecture:
- Backend creates K8s Job manifest for each training request
- K8s Job runs Training Service container (same image as local dev)
- Same Training Service code, same environment variables
- Implements TrainingManager abstract interface

Requirements:
- kubernetes Python client: pip install kubernetes
- Either in-cluster config (when running in K8s) or kubeconfig file
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

from kubernetes import client, config
from kubernetes.client.rest import ApiException

from app.core.training_manager import TrainingManager

logger = logging.getLogger(__name__)


class KubernetesTrainingManager(TrainingManager):
    """
    Manages training jobs via Kubernetes Jobs (Tier 1+).

    Features:
    - Create K8s Job manifest from TrainingJob parameters
    - Submit job to K8s API
    - Monitor job status via K8s API
    - Delete jobs (stop/cleanup)
    - Same environment variables as SubprocessTrainingManager
    """

    def __init__(self):
        logger.info("[KubernetesTrainingManager] Initializing")

        # Load K8s configuration
        try:
            # Try in-cluster config first (when running inside K8s)
            config.load_incluster_config()
            logger.info("[KubernetesTrainingManager] Using in-cluster config")
        except config.ConfigException:
            # Fall back to kubeconfig file (local development)
            try:
                config.load_kube_config()
                logger.info("[KubernetesTrainingManager] Using kubeconfig file")
            except config.ConfigException as e:
                logger.error(f"[KubernetesTrainingManager] Failed to load K8s config: {e}")
                raise RuntimeError(
                    "Could not configure Kubernetes client. "
                    "Ensure you're running in a K8s cluster or have a valid kubeconfig."
                )

        # Initialize K8s API clients
        self.batch_api = client.BatchV1Api()
        self.core_api = client.CoreV1Api()

        # Configuration from environment
        self.namespace = os.getenv("K8S_TRAINING_NAMESPACE", "training")
        self.docker_registry = os.getenv("DOCKER_REGISTRY", "ghcr.io/vision-ai")
        self.trainer_image_tag = os.getenv("TRAINER_IMAGE_TAG", "latest")

        # Resource defaults
        self.default_cpu_request = os.getenv("TRAINER_CPU_REQUEST", "2")
        self.default_cpu_limit = os.getenv("TRAINER_CPU_LIMIT", "4")
        self.default_memory_request = os.getenv("TRAINER_MEMORY_REQUEST", "4Gi")
        self.default_memory_limit = os.getenv("TRAINER_MEMORY_LIMIT", "8Gi")
        self.default_gpu_limit = os.getenv("TRAINER_GPU_LIMIT", "1")

        # Job settings
        self.job_ttl_seconds = int(os.getenv("JOB_TTL_SECONDS_AFTER_FINISHED", "3600"))
        self.job_active_deadline = int(os.getenv("JOB_ACTIVE_DEADLINE_SECONDS", "86400"))  # 24h
        self.job_backoff_limit = int(os.getenv("JOB_BACKOFF_LIMIT", "2"))

        # Image pull secret (for private registries)
        self.image_pull_secret = os.getenv("IMAGE_PULL_SECRET", None)

        # Storage configuration for trainer pods
        self.storage_env_vars = [
            "EXTERNAL_STORAGE_ENDPOINT",
            "EXTERNAL_STORAGE_ACCESS_KEY",
            "EXTERNAL_STORAGE_SECRET_KEY",
            "EXTERNAL_BUCKET_DATASETS",
            "INTERNAL_STORAGE_ENDPOINT",
            "INTERNAL_STORAGE_ACCESS_KEY",
            "INTERNAL_STORAGE_SECRET_KEY",
            "INTERNAL_BUCKET_CHECKPOINTS",
        ]

        logger.info(f"[KubernetesTrainingManager] Namespace: {self.namespace}")
        logger.info(f"[KubernetesTrainingManager] Registry: {self.docker_registry}")
        logger.info(f"[KubernetesTrainingManager] Image tag: {self.trainer_image_tag}")

    def _get_trainer_image(self, framework: str) -> str:
        """
        Get Docker image URI for a framework.

        Args:
            framework: Framework name (ultralytics, timm, huggingface)

        Returns:
            Full image URI (e.g., ghcr.io/vision-ai/trainer-ultralytics:latest)
        """
        return f"{self.docker_registry}/trainer-{framework}:{self.trainer_image_tag}"

    def _generate_job_name(self, job_type: str, job_id: int) -> str:
        """
        Generate unique K8s Job name.

        Args:
            job_type: Type of job (training, eval, inference, export)
            job_id: Job ID

        Returns:
            K8s-compatible job name (e.g., training-123-a1b2c3)
        """
        import hashlib
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        hash_suffix = hashlib.md5(f"{job_id}-{timestamp}".encode()).hexdigest()[:6]
        return f"{job_type}-{job_id}-{hash_suffix}"

    def _build_env_vars(self, env_dict: Dict[str, str]) -> List[client.V1EnvVar]:
        """
        Convert dict to K8s V1EnvVar list.

        Args:
            env_dict: Dictionary of environment variables

        Returns:
            List of V1EnvVar objects
        """
        env_vars = []
        for key, value in env_dict.items():
            if value is not None:
                env_vars.append(client.V1EnvVar(name=key, value=str(value)))
        return env_vars

    def _build_resource_requirements(
        self,
        cpu_request: str = None,
        cpu_limit: str = None,
        memory_request: str = None,
        memory_limit: str = None,
        gpu_limit: str = None,
    ) -> client.V1ResourceRequirements:
        """
        Build K8s resource requirements.

        Args:
            cpu_request: CPU request (e.g., "2")
            cpu_limit: CPU limit (e.g., "4")
            memory_request: Memory request (e.g., "4Gi")
            memory_limit: Memory limit (e.g., "8Gi")
            gpu_limit: GPU limit (e.g., "1")

        Returns:
            V1ResourceRequirements object
        """
        requests = {
            "cpu": cpu_request or self.default_cpu_request,
            "memory": memory_request or self.default_memory_request,
        }
        limits = {
            "cpu": cpu_limit or self.default_cpu_limit,
            "memory": memory_limit or self.default_memory_limit,
        }

        # Add GPU if specified
        gpu = gpu_limit or self.default_gpu_limit
        if gpu and gpu != "0":
            limits["nvidia.com/gpu"] = gpu
            requests["nvidia.com/gpu"] = gpu

        return client.V1ResourceRequirements(requests=requests, limits=limits)

    def _create_job_manifest(
        self,
        job_name: str,
        image: str,
        env_vars: List[client.V1EnvVar],
        command: List[str] = None,
        args: List[str] = None,
        resources: client.V1ResourceRequirements = None,
        labels: Dict[str, str] = None,
    ) -> client.V1Job:
        """
        Create K8s Job manifest.

        Args:
            job_name: Job name
            image: Container image
            env_vars: Environment variables
            command: Container command (entrypoint)
            args: Container arguments
            resources: Resource requirements
            labels: Job labels

        Returns:
            V1Job object
        """
        # Default labels
        default_labels = {
            "app": "vision-ai-trainer",
            "managed-by": "vision-ai-backend",
        }
        if labels:
            default_labels.update(labels)

        # Container definition
        container = client.V1Container(
            name="trainer",
            image=image,
            image_pull_policy="Always",
            command=command or ["python"],
            args=args or ["train.py"],
            env=env_vars,
            resources=resources or self._build_resource_requirements(),
            volume_mounts=[
                client.V1VolumeMount(name="tmp", mount_path="/tmp"),
                client.V1VolumeMount(name="workspace", mount_path="/workspace"),
            ],
        )

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            volumes=[
                client.V1Volume(name="tmp", empty_dir=client.V1EmptyDirVolumeSource()),
                client.V1Volume(name="workspace", empty_dir=client.V1EmptyDirVolumeSource()),
            ],
            # Tolerations for GPU nodes
            tolerations=[
                client.V1Toleration(
                    key="nvidia.com/gpu",
                    operator="Exists",
                    effect="NoSchedule",
                )
            ],
        )

        # Add image pull secret if configured
        if self.image_pull_secret:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=self.image_pull_secret)
            ]

        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=default_labels),
            spec=pod_spec,
        )

        # Job spec
        job_spec = client.V1JobSpec(
            template=pod_template,
            backoff_limit=self.job_backoff_limit,
            ttl_seconds_after_finished=self.job_ttl_seconds,
            active_deadline_seconds=self.job_active_deadline,
        )

        # Job
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(name=job_name, labels=default_labels),
            spec=job_spec,
        )

        return job

    async def start_training(
        self,
        job_id: int,
        framework: str,
        model_name: str,
        dataset_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
        snapshot_id: str = None,
        dataset_version_hash: str = None,
    ) -> Dict[str, Any]:
        """
        Start training by creating a K8s Job.

        Args:
            job_id: Training job ID
            framework: Framework name (ultralytics, timm, huggingface)
            model_name: Model name to train
            dataset_s3_uri: S3 URI of dataset
            callback_url: Backend API callback URL
            config: Training configuration dictionary
            snapshot_id: Dataset snapshot ID (for caching)
            dataset_version_hash: Dataset version hash (for caching)

        Returns:
            Dict containing K8s job metadata

        Raises:
            ApiException: If K8s API call fails
        """
        logger.info(f"[KubernetesTrainingManager] Starting training job {job_id}")
        logger.info(f"[KubernetesTrainingManager]   Framework: {framework}")
        logger.info(f"[KubernetesTrainingManager]   Model: {model_name}")
        logger.info(f"[KubernetesTrainingManager]   Dataset: {dataset_s3_uri}")

        try:
            # Generate job name
            k8s_job_name = self._generate_job_name("training", job_id)

            # Get trainer image
            image = self._get_trainer_image(framework)
            logger.info(f"[KubernetesTrainingManager]   Image: {image}")

            # Extract dataset_id from S3 URI
            dataset_id_match = re.search(r"/datasets/([^/]+)/?$", dataset_s3_uri)
            dataset_id = dataset_id_match.group(1) if dataset_id_match else ""

            # Build environment variables (same as SubprocessTrainingManager)
            env_dict = {
                "JOB_ID": str(job_id),
                "CALLBACK_URL": callback_url,
                "MODEL_NAME": model_name,
                "TASK_TYPE": config.get("task", "detection"),
                "FRAMEWORK": framework,
                "DATASET_ID": dataset_id,
                "DATASET_S3_URI": dataset_s3_uri,
                # Basic training parameters
                "EPOCHS": str(config.get("epochs", 100)),
                "BATCH_SIZE": str(config.get("batch", 16)),
                "LEARNING_RATE": str(config.get("learning_rate", 0.01)),
                "IMGSZ": str(config.get("imgsz", 640)),
                "DEVICE": str(config.get("device", "0")),
            }

            # Dataset caching parameters
            if snapshot_id:
                env_dict["SNAPSHOT_ID"] = snapshot_id
            if dataset_version_hash:
                env_dict["DATASET_VERSION_HASH"] = dataset_version_hash

            # CONFIG JSON (all training parameters)
            config_json = {
                "epochs": config.get("epochs", 100),
                "batch": config.get("batch", 16),
                "learning_rate": config.get("learning_rate", 0.01),
                "imgsz": config.get("imgsz", 640),
                "device": config.get("device", "0"),
                "advanced_config": config.get("advanced_config", {}),
                "primary_metric": config.get("primary_metric"),
                "primary_metric_mode": config.get("primary_metric_mode", "max"),
                "split_config": config.get("split_config"),
            }
            config_json = {k: v for k, v in config_json.items() if v is not None}
            env_dict["CONFIG"] = json.dumps(config_json)

            # Storage environment variables
            for var in self.storage_env_vars:
                if var in os.environ:
                    env_dict[var] = os.environ[var]

            env_vars = self._build_env_vars(env_dict)

            # Labels for tracking
            labels = {
                "job-id": str(job_id),
                "job-type": "training",
                "framework": framework,
            }

            # Create job manifest
            job = self._create_job_manifest(
                job_name=k8s_job_name,
                image=image,
                env_vars=env_vars,
                args=["train.py", "--log-level", "INFO"],
                labels=labels,
            )

            # Submit job to K8s
            created_job = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )

            logger.info(
                f"[KubernetesTrainingManager] Job {job_id} created: "
                f"{created_job.metadata.name} in namespace {self.namespace}"
            )

            return {
                "job_id": job_id,
                "k8s_job_name": created_job.metadata.name,
                "k8s_namespace": self.namespace,
                "image": image,
                "status": "submitted",
                "created_at": created_job.metadata.creation_timestamp.isoformat()
                if created_job.metadata.creation_timestamp
                else datetime.utcnow().isoformat(),
            }

        except ApiException as e:
            logger.error(f"[KubernetesTrainingManager] K8s API error: {e}")
            raise RuntimeError(f"Failed to create K8s job: {e.reason}")
        except Exception as e:
            logger.error(f"[KubernetesTrainingManager] Failed to start job {job_id}: {e}")
            raise

    def stop_training(self, job_id: int) -> bool:
        """
        Stop a running training job by deleting the K8s Job.

        Args:
            job_id: Training job ID

        Returns:
            True if successfully stopped, False otherwise
        """
        logger.info(f"[KubernetesTrainingManager] Stopping job {job_id}")

        try:
            # Find job by label
            jobs = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"job-id={job_id},job-type=training",
            )

            if not jobs.items:
                logger.warning(f"[KubernetesTrainingManager] No K8s job found for job_id {job_id}")
                return False

            # Delete all matching jobs (should be only one)
            for job in jobs.items:
                logger.info(
                    f"[KubernetesTrainingManager] Deleting job: {job.metadata.name}"
                )
                self.batch_api.delete_namespaced_job(
                    name=job.metadata.name,
                    namespace=self.namespace,
                    body=client.V1DeleteOptions(
                        propagation_policy="Foreground"  # Delete pods too
                    ),
                )

            logger.info(f"[KubernetesTrainingManager] Job {job_id} stopped")
            return True

        except ApiException as e:
            logger.error(f"[KubernetesTrainingManager] Failed to stop job {job_id}: {e}")
            return False

    def get_training_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """
        Get status of a training job from K8s.

        Args:
            job_id: Training job ID

        Returns:
            Status dictionary or None if not found
        """
        try:
            # Find job by label
            jobs = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"job-id={job_id},job-type=training",
            )

            if not jobs.items:
                return None

            job = jobs.items[0]
            status = job.status

            # Determine job state
            if status.succeeded and status.succeeded > 0:
                state = "completed"
            elif status.failed and status.failed > 0:
                state = "failed"
            elif status.active and status.active > 0:
                state = "running"
            else:
                state = "pending"

            return {
                "job_id": job_id,
                "k8s_job_name": job.metadata.name,
                "status": state,
                "active": status.active or 0,
                "succeeded": status.succeeded or 0,
                "failed": status.failed or 0,
                "start_time": status.start_time.isoformat() if status.start_time else None,
                "completion_time": status.completion_time.isoformat()
                if status.completion_time
                else None,
            }

        except ApiException as e:
            logger.error(
                f"[KubernetesTrainingManager] Failed to get status for job {job_id}: {e}"
            )
            return None

    async def start_evaluation(
        self,
        test_run_id: int,
        training_job_id: Optional[int],
        framework: str,
        checkpoint_s3_uri: str,
        dataset_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start evaluation by creating a K8s Job.

        Args:
            test_run_id: Test run ID
            training_job_id: Original training job ID (optional)
            framework: Framework name
            checkpoint_s3_uri: S3 URI to checkpoint
            dataset_s3_uri: S3 URI to test dataset
            callback_url: Backend API callback URL
            config: Evaluation configuration

        Returns:
            Dict containing K8s job metadata
        """
        logger.info(f"[KubernetesTrainingManager] Starting evaluation {test_run_id}")

        try:
            k8s_job_name = self._generate_job_name("eval", test_run_id)
            image = self._get_trainer_image(framework)

            env_dict = {
                "TEST_RUN_ID": str(test_run_id),
                "CHECKPOINT_S3_URI": checkpoint_s3_uri,
                "DATASET_S3_URI": dataset_s3_uri,
                "CALLBACK_URL": callback_url,
                "CONFIG": json.dumps(config),
            }

            if training_job_id:
                env_dict["TRAINING_JOB_ID"] = str(training_job_id)

            # Storage environment variables
            for var in self.storage_env_vars:
                if var in os.environ:
                    env_dict[var] = os.environ[var]

            env_vars = self._build_env_vars(env_dict)

            labels = {
                "job-id": str(test_run_id),
                "job-type": "evaluation",
                "framework": framework,
            }

            job = self._create_job_manifest(
                job_name=k8s_job_name,
                image=image,
                env_vars=env_vars,
                args=["evaluate.py", "--log-level", "INFO"],
                labels=labels,
            )

            created_job = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )

            logger.info(
                f"[KubernetesTrainingManager] Evaluation job created: {created_job.metadata.name}"
            )

            return {
                "test_run_id": test_run_id,
                "k8s_job_name": created_job.metadata.name,
                "k8s_namespace": self.namespace,
                "status": "submitted",
            }

        except ApiException as e:
            logger.error(f"[KubernetesTrainingManager] K8s API error: {e}")
            raise RuntimeError(f"Failed to create evaluation job: {e.reason}")

    async def start_inference(
        self,
        inference_job_id: int,
        training_job_id: Optional[int],
        framework: str,
        checkpoint_s3_uri: str,
        images_s3_uri: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start inference by creating a K8s Job.

        Args:
            inference_job_id: Inference job ID
            training_job_id: Original training job ID (optional)
            framework: Framework name
            checkpoint_s3_uri: S3 URI to checkpoint
            images_s3_uri: S3 URI to input images
            callback_url: Backend API callback URL
            config: Inference configuration

        Returns:
            Dict containing K8s job metadata
        """
        logger.info(f"[KubernetesTrainingManager] Starting inference {inference_job_id}")

        try:
            k8s_job_name = self._generate_job_name("inference", inference_job_id)
            image = self._get_trainer_image(framework)

            env_dict = {
                "INFERENCE_JOB_ID": str(inference_job_id),
                "CHECKPOINT_S3_URI": checkpoint_s3_uri,
                "IMAGES_S3_URI": images_s3_uri,
                "CALLBACK_URL": callback_url,
                "CONFIG": json.dumps(config),
            }

            if training_job_id:
                env_dict["TRAINING_JOB_ID"] = str(training_job_id)

            # Storage environment variables
            for var in self.storage_env_vars:
                if var in os.environ:
                    env_dict[var] = os.environ[var]

            env_vars = self._build_env_vars(env_dict)

            labels = {
                "job-id": str(inference_job_id),
                "job-type": "inference",
                "framework": framework,
            }

            job = self._create_job_manifest(
                job_name=k8s_job_name,
                image=image,
                env_vars=env_vars,
                args=["predict.py", "--log-level", "INFO"],
                labels=labels,
            )

            created_job = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )

            logger.info(
                f"[KubernetesTrainingManager] Inference job created: {created_job.metadata.name}"
            )

            return {
                "inference_job_id": inference_job_id,
                "k8s_job_name": created_job.metadata.name,
                "k8s_namespace": self.namespace,
                "status": "submitted",
            }

        except ApiException as e:
            logger.error(f"[KubernetesTrainingManager] K8s API error: {e}")
            raise RuntimeError(f"Failed to create inference job: {e.reason}")

    async def start_export(
        self,
        export_job_id: int,
        training_job_id: int,
        framework: str,
        checkpoint_s3_uri: str,
        export_format: str,
        callback_url: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start model export by creating a K8s Job.

        Args:
            export_job_id: Export job ID
            training_job_id: Original training job ID
            framework: Framework name
            checkpoint_s3_uri: S3 URI to checkpoint
            export_format: Export format (onnx, tensorrt, etc.)
            callback_url: Backend API callback URL
            config: Export configuration

        Returns:
            Dict containing K8s job metadata
        """
        logger.info(
            f"[KubernetesTrainingManager] Starting export {export_job_id} "
            f"(format: {export_format})"
        )

        try:
            k8s_job_name = self._generate_job_name("export", export_job_id)
            image = self._get_trainer_image(framework)

            env_dict = {
                "EXPORT_JOB_ID": str(export_job_id),
                "TRAINING_JOB_ID": str(training_job_id),
                "CHECKPOINT_S3_URI": checkpoint_s3_uri,
                "EXPORT_FORMAT": export_format,
                "CALLBACK_URL": callback_url,
                "CONFIG": json.dumps(config),
            }

            # Storage environment variables
            for var in self.storage_env_vars:
                if var in os.environ:
                    env_dict[var] = os.environ[var]

            env_vars = self._build_env_vars(env_dict)

            labels = {
                "job-id": str(export_job_id),
                "job-type": "export",
                "framework": framework,
                "export-format": export_format,
            }

            job = self._create_job_manifest(
                job_name=k8s_job_name,
                image=image,
                env_vars=env_vars,
                args=["export.py", "--log-level", "INFO"],
                labels=labels,
            )

            created_job = self.batch_api.create_namespaced_job(
                namespace=self.namespace,
                body=job,
            )

            logger.info(
                f"[KubernetesTrainingManager] Export job created: {created_job.metadata.name}"
            )

            return {
                "export_job_id": export_job_id,
                "k8s_job_name": created_job.metadata.name,
                "k8s_namespace": self.namespace,
                "export_format": export_format,
                "status": "submitted",
            }

        except ApiException as e:
            logger.error(f"[KubernetesTrainingManager] K8s API error: {e}")
            raise RuntimeError(f"Failed to create export job: {e.reason}")

    def cleanup_resources(self, job_id: int) -> None:
        """
        Clean up K8s resources for a job.

        Deletes all jobs with matching job-id label.

        Args:
            job_id: Job ID to clean up
        """
        logger.info(f"[KubernetesTrainingManager] Cleaning up resources for job {job_id}")

        try:
            # Find all jobs with this job_id (training, eval, inference, export)
            jobs = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"job-id={job_id}",
            )

            for job in jobs.items:
                logger.info(
                    f"[KubernetesTrainingManager] Deleting job: {job.metadata.name}"
                )
                try:
                    self.batch_api.delete_namespaced_job(
                        name=job.metadata.name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(propagation_policy="Foreground"),
                    )
                except ApiException as e:
                    if e.status != 404:  # Ignore "not found" errors
                        logger.warning(
                            f"[KubernetesTrainingManager] Failed to delete job "
                            f"{job.metadata.name}: {e}"
                        )

            logger.info(f"[KubernetesTrainingManager] Cleanup completed for job {job_id}")

        except ApiException as e:
            logger.error(
                f"[KubernetesTrainingManager] Failed to cleanup job {job_id}: {e}"
            )

    def get_job_logs(
        self, job_id: int, job_type: str = "training", tail_lines: int = 100
    ) -> Optional[str]:
        """
        Get logs from a K8s Job's pod.

        Args:
            job_id: Job ID
            job_type: Job type (training, evaluation, inference, export)
            tail_lines: Number of lines to return from the end

        Returns:
            Log string or None if not found
        """
        try:
            # Find job
            jobs = self.batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"job-id={job_id},job-type={job_type}",
            )

            if not jobs.items:
                return None

            job_name = jobs.items[0].metadata.name

            # Find pod for this job
            pods = self.core_api.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                return None

            pod_name = pods.items[0].metadata.name

            # Get logs
            logs = self.core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                container="trainer",
                tail_lines=tail_lines,
            )

            return logs

        except ApiException as e:
            logger.error(
                f"[KubernetesTrainingManager] Failed to get logs for job {job_id}: {e}"
            )
            return None
