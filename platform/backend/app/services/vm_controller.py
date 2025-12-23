"""
Kubernetes Job Controller for Training Workloads.

Manages the lifecycle of training jobs in Kubernetes cluster.
"""

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from typing import Dict, Any, Optional
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingJobConfig:
    """Configuration for creating K8s training job"""

    def __init__(
        self,
        job_id: int,
        framework: str,
        task_type: str,
        model_name: str,
        dataset_path: str,
        dataset_format: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: str = "adam",
        num_classes: Optional[int] = None,
        project_id: Optional[int] = None,
        image_size: Optional[int] = None,
        pretrained: bool = True,
        advanced_config: Optional[Dict[str, Any]] = None,
        resources: Optional[Dict[str, Any]] = None,
        executor: Optional[str] = None,
    ):
        self.job_id = job_id
        self.framework = framework
        self.task_type = task_type
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.project_id = project_id
        self.image_size = image_size
        self.pretrained = pretrained
        self.advanced_config = advanced_config or {}
        self.resources = resources or {
            "gpu": 1,
            "memory": "16Gi",
            "cpu": 4,
            "memory_request": "8Gi",
            "cpu_request": 2,
        }
        self.executor = executor


class VMController:
    """Kubernetes Job controller for training workloads"""

    def __init__(self, namespace: str = "training"):
        """
        Initialize K8s client

        Args:
            namespace: Kubernetes namespace for training jobs
        """
        self.namespace = namespace

        # Initialize Kubernetes client
        try:
            # Try in-cluster config first (production)
            config.load_incluster_config()
            logger.info("[K8S] Using in-cluster configuration")
        except:
            # Fallback to local kubeconfig (development)
            try:
                config.load_kube_config()
                logger.info("[K8S] Using local kubeconfig")
            except Exception as e:
                logger.warning(f"[K8S] Failed to load kubeconfig: {e}")
                logger.warning("[K8S] K8s client not initialized - will fail on job creation")

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Ensure namespace exists
        self._ensure_namespace()

    def _ensure_namespace(self):
        """Create namespace if it doesn't exist"""
        try:
            self.core_v1.read_namespace(name=self.namespace)
            logger.info(f"[K8S] Namespace '{self.namespace}' exists")
        except ApiException as e:
            if e.status == 404:
                # Create namespace
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=self.namespace)
                )
                self.core_v1.create_namespace(body=namespace_manifest)
                logger.info(f"[K8S] Created namespace '{self.namespace}'")
            else:
                logger.error(f"[K8S] Failed to check namespace: {e}")

    def create_training_job(self, job_config: TrainingJobConfig) -> str:
        """
        Create Kubernetes Job for training

        Args:
            job_config: Training job configuration

        Returns:
            job_name: Kubernetes Job name (e.g., "training-job-123")

        Raises:
            ApiException: If job creation fails
        """
        job_name = f"training-job-{job_config.job_id}"

        logger.info(f"[K8S] Creating training job: {job_name}")
        logger.info(f"[K8S]   Framework: {job_config.framework}")
        logger.info(f"[K8S]   Model: {job_config.model_name}")
        logger.info(f"[K8S]   Dataset: {job_config.dataset_path}")

        # Select Docker image based on framework
        image = self._get_image(job_config.framework)
        logger.info(f"[K8S]   Image: {image}")

        # Build command arguments for train.py
        args = self._build_training_args(job_config)

        # Build Job manifest
        job_manifest = self._build_job_manifest(
            job_name=job_name,
            image=image,
            args=args,
            resources=job_config.resources,
            labels={
                "app": "training-job",
                "job-id": str(job_config.job_id),
                "framework": job_config.framework,
                "project-id": str(job_config.project_id) if job_config.project_id else "none",
            },
        )

        # Create Job in Kubernetes
        try:
            job = self.batch_v1.create_namespaced_job(
                namespace=self.namespace, body=job_manifest
            )
            logger.info(f"[K8S] ✓ Created Job: {job_name}")
            return job_name

        except ApiException as e:
            logger.error(f"[K8S] ✗ Failed to create Job: {e}")
            raise

    def get_job_status(self, job_name: str) -> str:
        """
        Get Kubernetes Job status

        Args:
            job_name: Kubernetes Job name

        Returns:
            status: "pending" | "running" | "completed" | "failed" | "not_found"
        """
        try:
            job = self.batch_v1.read_namespaced_job_status(
                name=job_name, namespace=self.namespace
            )

            # Check job status
            if job.status.succeeded and job.status.succeeded > 0:
                return "completed"
            elif job.status.failed and job.status.failed > 0:
                return "failed"
            elif job.status.active and job.status.active > 0:
                return "running"
            else:
                return "pending"

        except ApiException as e:
            if e.status == 404:
                return "not_found"
            logger.error(f"[K8S] Error getting job status: {e}")
            raise

    def get_job_logs(self, job_name: str, tail_lines: int = 100) -> str:
        """
        Get logs from training pod

        Args:
            job_name: Kubernetes Job name
            tail_lines: Number of lines to return from end of log

        Returns:
            logs: Pod logs (stdout/stderr)
        """
        try:
            # Find pod for this job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={job_name}"
            )

            if not pods.items:
                return ""

            pod_name = pods.items[0].metadata.name

            # Get logs
            logs = self.core_v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                tail_lines=tail_lines,
                timestamps=True,
            )

            return logs

        except ApiException as e:
            logger.error(f"[K8S] Failed to get logs: {e}")
            return ""

    def delete_job(self, job_name: str):
        """
        Delete Kubernetes Job and associated pods

        Args:
            job_name: Kubernetes Job name
        """
        try:
            self.batch_v1.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                propagation_policy="Foreground",  # Delete pods first
            )
            logger.info(f"[K8S] ✓ Deleted Job: {job_name}")

        except ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                logger.error(f"[K8S] Failed to delete Job: {e}")
                raise

    # Helper methods

    def _get_image(self, framework: str) -> str:
        """Get Docker image for framework"""
        registry = os.getenv("DOCKER_REGISTRY", "ghcr.io/myorg")
        version = os.getenv("TRAINER_IMAGE_VERSION", "v1.0")

        images = {
            "ultralytics": f"{registry}/trainer-ultralytics:{version}",
            "timm": f"{registry}/trainer-timm:{version}",
            "huggingface": f"{registry}/trainer-huggingface:{version}",
            "mmdet": f"{registry}/trainer-mmdet:{version}",
            "mmpretrain": f"{registry}/trainer-mmpretrain:{version}",
            "mmseg": f"{registry}/trainer-mmseg:{version}",
            "mmyolo": f"{registry}/trainer-mmyolo:{version}",
        }

        return images.get(framework, images["ultralytics"])

    def _build_training_args(self, job_config: TrainingJobConfig) -> list:
        """Build command-line arguments for train.py"""
        args = [
            f"--framework={job_config.framework}",
            f"--task_type={job_config.task_type}",
            f"--model_name={job_config.model_name}",
            f"--dataset_path={job_config.dataset_path}",
            f"--dataset_format={job_config.dataset_format}",
            f"--epochs={job_config.epochs}",
            f"--batch_size={job_config.batch_size}",
            f"--learning_rate={job_config.learning_rate}",
            f"--optimizer={job_config.optimizer}",
            f"--output_dir=/workspace/output",
            f"--job_id={job_config.job_id}",
        ]

        # Optional args
        if job_config.num_classes:
            args.append(f"--num_classes={job_config.num_classes}")

        if job_config.project_id:
            args.append(f"--project_id={job_config.project_id}")

        if job_config.image_size:
            args.append(f"--image_size={job_config.image_size}")

        if job_config.pretrained:
            args.append("--pretrained")

        if job_config.advanced_config:
            # Serialize advanced_config as JSON string
            config_json = json.dumps(job_config.advanced_config)
            # Escape quotes for shell safety
            config_json = config_json.replace('"', '\\"')
            args.append(f'--advanced_config={config_json}')

        return args

    def _build_job_manifest(
        self,
        job_name: str,
        image: str,
        args: list,
        resources: Dict[str, Any],
        labels: Dict[str, str],
    ) -> client.V1Job:
        """Build Kubernetes Job manifest"""

        # Container spec
        container = client.V1Container(
            name="trainer",
            image=image,
            image_pull_policy="IfNotPresent",
            command=["python", "train.py"],
            args=args,
            env=[
                # R2 Credentials (from Secret)
                client.V1EnvVar(
                    name="AWS_S3_ENDPOINT_URL",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials", key="endpoint", optional=True
                        )
                    ),
                ),
                client.V1EnvVar(
                    name="AWS_ACCESS_KEY_ID",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials", key="access-key", optional=True
                        )
                    ),
                ),
                client.V1EnvVar(
                    name="AWS_SECRET_ACCESS_KEY",
                    value_from=client.V1EnvVarSource(
                        secret_key_ref=client.V1SecretKeySelector(
                            name="r2-credentials", key="secret-key", optional=True
                        )
                    ),
                ),
                # Backend API (from ConfigMap)
                client.V1EnvVar(
                    name="BACKEND_API_URL",
                    value_from=client.V1EnvVarSource(
                        config_map_key_ref=client.V1ConfigMapKeySelector(
                            name="backend-config", key="api-url", optional=True
                        )
                    ),
                ),
                # MLflow
                client.V1EnvVar(
                    name="MLFLOW_TRACKING_URI", value="http://mlflow-service:5000"
                ),
                # CUDA
                client.V1EnvVar(name="CUDA_VISIBLE_DEVICES", value="0"),
            ],
            resources=client.V1ResourceRequirements(
                limits={
                    "nvidia.com/gpu": str(resources.get("gpu", 1)),
                    "memory": resources.get("memory", "16Gi"),
                    "cpu": str(resources.get("cpu", 4)),
                },
                requests={
                    "nvidia.com/gpu": str(resources.get("gpu", 1)),
                    "memory": resources.get("memory_request", "8Gi"),
                    "cpu": str(resources.get("cpu_request", 2)),
                },
            ),
            volume_mounts=[
                client.V1VolumeMount(name="workspace", mount_path="/workspace"),
                client.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
            ],
        )

        # Pod template
        pod_template = client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels=labels),
            spec=client.V1PodSpec(
                restart_policy="Never",
                containers=[container],
                node_selector={"accelerator": "nvidia-gpu"},
                tolerations=[
                    client.V1Toleration(
                        key="nvidia.com/gpu",
                        operator="Exists",
                        effect="NoSchedule",
                    )
                ],
                volumes=[
                    client.V1Volume(
                        name="workspace",
                        empty_dir=client.V1EmptyDirVolumeSource(),
                    ),
                    client.V1Volume(
                        name="dshm",
                        empty_dir=client.V1EmptyDirVolumeSource(
                            medium="Memory", size_limit="2Gi"
                        ),
                    ),
                ],
            ),
        )

        # Job spec
        job_spec = client.V1JobSpec(
            ttl_seconds_after_finished=86400,  # 24 hours
            backoff_limit=3,  # Max 3 retries
            active_deadline_seconds=86400,  # Max 24h runtime
            template=pod_template,
        )

        # Job manifest
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name, namespace=self.namespace, labels=labels
            ),
            spec=job_spec,
        )

        return job
