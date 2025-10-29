"""Database models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)

    company = Column(String(100), nullable=True)
    company_custom = Column(String(255), nullable=True)
    division = Column(String(100), nullable=True)
    division_custom = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)

    phone_number = Column(String(50), nullable=True)
    bio = Column(Text, nullable=True)

    system_role = Column(String(50), nullable=False, default='guest')
    is_active = Column(Boolean, nullable=False, default=True)
    badge_color = Column(String(20), nullable=True)  # Avatar badge color (e.g., 'blue', 'green', 'purple')

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    owned_projects = relationship("Project", back_populates="owner", foreign_keys="[Project.user_id]")
    sessions = relationship("Session", back_populates="user")
    created_training_jobs = relationship("TrainingJob", back_populates="creator", foreign_keys="[TrainingJob.created_by]")
    project_memberships = relationship("ProjectMember", back_populates="user", foreign_keys="[ProjectMember.user_id]")
    invited_members = relationship("ProjectMember", back_populates="inviter", foreign_keys="[ProjectMember.invited_by]")


class ProjectMember(Base):
    """Project member model for collaboration."""

    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False, default='member')
    invited_by = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships", foreign_keys=[user_id])
    inviter = relationship("User", back_populates="invited_members", foreign_keys=[invited_by])


class Project(Base):
    """Project model for organizing experiments."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    task_type = Column(String(50), nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="owned_projects", foreign_keys=[user_id])
    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    experiments = relationship("TrainingJob", back_populates="project", cascade="all, delete-orphan")


class Session(Base):
    """Chat session model."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    state = Column(String(50), nullable=False, default="initial", index=True)
    temp_data = Column(JSON, nullable=False, default={})

    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    training_jobs = relationship("TrainingJob", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    """Chat message model."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="messages")


class TrainingJob(Base):
    """Training job model (also serves as Experiment)."""

    __tablename__ = "training_jobs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    experiment_name = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    mlflow_experiment_id = Column(String(100), nullable=True)  # MLflow experiment ID
    mlflow_run_id = Column(String(100), nullable=True)  # MLflow run ID for this training

    framework = Column(String(50), nullable=False, default="timm")
    model_name = Column(String(100), nullable=False)
    task_type = Column(String(50), nullable=False)
    num_classes = Column(Integer, nullable=True)
    dataset_path = Column(String(500), nullable=False)
    dataset_format = Column(String(50), nullable=False, default="imagefolder")
    output_dir = Column(String(500), nullable=False)

    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)

    # Advanced configuration (JSON field for optimizer, scheduler, augmentation, etc.)
    advanced_config = Column(JSON, nullable=True)

    status = Column(String(20), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    process_id = Column(Integer, nullable=True)

    final_accuracy = Column(Float, nullable=True)
    best_checkpoint_path = Column(String(500), nullable=True)

    # Primary metric configuration
    primary_metric = Column(String(100), nullable=True, default="loss")  # Metric name to optimize (e.g., 'accuracy', 'mAP50', 'f1_score')
    primary_metric_mode = Column(String(10), nullable=True, default="min")  # 'min' or 'max'

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    creator = relationship("User", back_populates="created_training_jobs", foreign_keys=[created_by])
    session = relationship("Session", back_populates="training_jobs")
    project = relationship("Project", back_populates="experiments")
    metrics = relationship("TrainingMetric", back_populates="job", cascade="all, delete-orphan")
    logs = relationship("TrainingLog", back_populates="job", cascade="all, delete-orphan")
    validation_results = relationship("ValidationResult", back_populates="job", cascade="all, delete-orphan")
    test_runs = relationship("TestRun", back_populates="training_job", cascade="all, delete-orphan")
    inference_jobs = relationship("InferenceJob", back_populates="training_job", cascade="all, delete-orphan")


class TrainingMetric(Base):
    """Training metric model."""

    __tablename__ = "training_metrics"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    epoch = Column(Integer, nullable=False)
    step = Column(Integer, nullable=True)

    loss = Column(Float, nullable=True)
    accuracy = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)

    extra_metrics = Column(JSON, nullable=True)
    checkpoint_path = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="metrics")


class TrainingLog(Base):
    """Training log model for capturing stdout/stderr."""

    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False)

    log_type = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("TrainingJob", back_populates="logs")


class ValidationResult(Base):
    """Task-agnostic validation result model.

    Stores validation metrics for any computer vision task (classification,
    detection, segmentation, pose estimation, etc.) using flexible JSON fields.
    """

    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False, index=True)

    # Task identification
    task_type = Column(String(50), nullable=False, index=True)

    # Primary metric (task-specific)
    primary_metric_value = Column(Float, nullable=True)
    primary_metric_name = Column(String(100), nullable=True)
    overall_loss = Column(Float, nullable=True)

    # Task-specific metrics stored as JSON
    metrics = Column(JSON, nullable=True)  # Overall metrics dict
    per_class_metrics = Column(JSON, nullable=True)  # Per-class/category metrics

    # Visualization data (task-specific)
    confusion_matrix = Column(JSON, nullable=True)  # Classification
    pr_curves = Column(JSON, nullable=True)  # Detection, Segmentation
    class_names = Column(JSON, nullable=True)  # Class/category labels
    visualization_data = Column(JSON, nullable=True)  # Additional viz data

    # Sample images for UI display
    sample_correct_images = Column(JSON, nullable=True)
    sample_incorrect_images = Column(JSON, nullable=True)

    # Checkpoint path for this validation epoch
    checkpoint_path = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    job = relationship("TrainingJob", back_populates="validation_results")
    image_results = relationship("ValidationImageResult", back_populates="validation_result", cascade="all, delete-orphan")


class ValidationImageResult(Base):
    """Task-agnostic image-level validation result model.

    Stores per-image validation results with fields supporting all task types.
    Only relevant fields are populated based on task_type.
    """

    __tablename__ = "validation_image_results"

    id = Column(Integer, primary_key=True, index=True)
    validation_result_id = Column(Integer, ForeignKey("validation_results.id"), nullable=False, index=True)
    job_id = Column(Integer, ForeignKey("training_jobs.id"), nullable=False, index=True)
    epoch = Column(Integer, nullable=False, index=True)

    # Image information
    image_path = Column(String(500), nullable=True)  # Made nullable - path may not be available during training
    image_name = Column(String(200), nullable=False, index=True)
    image_index = Column(Integer, nullable=True)

    # Classification fields
    true_label = Column(String(100), nullable=True)
    true_label_id = Column(Integer, nullable=True)
    predicted_label = Column(String(100), nullable=True)
    predicted_label_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)  # List of (label, confidence)

    # Object Detection fields
    true_boxes = Column(JSON, nullable=True)  # List of ground truth bounding boxes
    predicted_boxes = Column(JSON, nullable=True)  # List of predicted bounding boxes

    # Segmentation fields
    true_mask_path = Column(String(500), nullable=True)
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose Estimation fields
    true_keypoints = Column(JSON, nullable=True)  # Ground truth keypoints
    predicted_keypoints = Column(JSON, nullable=True)  # Predicted keypoints

    # Common metrics
    is_correct = Column(Boolean, nullable=False, default=False, index=True)
    iou = Column(Float, nullable=True)  # IoU for detection/segmentation
    oks = Column(Float, nullable=True)  # Object Keypoint Similarity for pose

    # Extra data for task-specific needs
    extra_data = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    validation_result = relationship("ValidationResult", back_populates="image_results")


# ========== Test Run Models ==========

class TestRun(Base):
    """
    Test run on a labeled dataset after training.

    Similar to validation but runs on a separate test set for final evaluation.
    """

    __tablename__ = "test_runs"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    checkpoint_path = Column(String(500), nullable=False)
    dataset_path = Column(String(500), nullable=False)
    dataset_split = Column(String(20), default="test")

    # Status
    status = Column(String(20), nullable=False, index=True)  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)

    # Task info
    task_type = Column(String(50), nullable=False)
    primary_metric_name = Column(String(50), nullable=True)
    primary_metric_value = Column(Float, nullable=True)

    # Metrics (task-agnostic JSON)
    overall_loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    per_class_metrics = Column(JSON, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)

    # Metadata
    class_names = Column(JSON, nullable=True)
    total_images = Column(Integer, default=0)
    inference_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="test_runs")
    image_results = relationship("TestImageResult", back_populates="test_run", cascade="all, delete-orphan")


class TestImageResult(Base):
    """
    Per-image test result with predictions and ground truth.

    Similar to ValidationImageResult but for test set evaluation.
    """

    __tablename__ = "test_image_results"

    id = Column(Integer, primary_key=True, index=True)
    test_run_id = Column(Integer, ForeignKey("test_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    # Image information
    image_path = Column(String(500), nullable=True)
    image_name = Column(String(200), nullable=False)
    image_index = Column(Integer, nullable=True)

    # Classification fields
    true_label = Column(String(100), nullable=True)
    true_label_id = Column(Integer, nullable=True, index=True)
    predicted_label = Column(String(100), nullable=True)
    predicted_label_id = Column(Integer, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)

    # Detection fields
    true_boxes = Column(JSON, nullable=True)
    predicted_boxes = Column(JSON, nullable=True)

    # Segmentation fields
    true_mask_path = Column(String(500), nullable=True)
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose fields
    true_keypoints = Column(JSON, nullable=True)
    predicted_keypoints = Column(JSON, nullable=True)

    # Metrics
    is_correct = Column(Boolean, nullable=False, default=False, index=True)
    iou = Column(Float, nullable=True)
    oks = Column(Float, nullable=True)

    # Performance
    inference_time_ms = Column(Float, nullable=True)
    preprocessing_time_ms = Column(Float, default=0.0)
    postprocessing_time_ms = Column(Float, default=0.0)

    # Extra data
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    test_run = relationship("TestRun", back_populates="image_results")
    job = relationship("TrainingJob", foreign_keys=[training_job_id])


# ========== Inference Models ==========

class InferenceJob(Base):
    """
    Inference job on unlabeled images (production use case).

    No ground truth, no metrics - only predictions and visualizations.
    """

    __tablename__ = "inference_jobs"

    id = Column(Integer, primary_key=True, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    checkpoint_path = Column(String(500), nullable=False)

    # Input
    inference_type = Column(String(20), nullable=False)  # single, batch, dataset
    input_data = Column(JSON, nullable=True)

    # Status
    status = Column(String(20), nullable=False, index=True)
    error_message = Column(Text, nullable=True)

    # Task info
    task_type = Column(String(50), nullable=False)

    # Performance metrics
    total_images = Column(Integer, default=0)
    total_inference_time_ms = Column(Float, nullable=True)
    avg_inference_time_ms = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    training_job = relationship("TrainingJob", back_populates="inference_jobs")
    results = relationship("InferenceResult", back_populates="inference_job", cascade="all, delete-orphan")


class InferenceResult(Base):
    """
    Per-image inference result (no ground truth).

    Used for production inference on unlabeled images.
    """

    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    inference_job_id = Column(Integer, ForeignKey("inference_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    # Image information
    image_path = Column(String(500), nullable=False)
    image_name = Column(String(200), nullable=False)
    image_index = Column(Integer, nullable=True)

    # Classification predictions
    predicted_label = Column(String(100), nullable=True)
    predicted_label_id = Column(Integer, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    top5_predictions = Column(JSON, nullable=True)

    # Detection predictions
    predicted_boxes = Column(JSON, nullable=True)

    # Segmentation predictions
    predicted_mask_path = Column(String(500), nullable=True)

    # Pose predictions
    predicted_keypoints = Column(JSON, nullable=True)

    # Performance
    inference_time_ms = Column(Float, nullable=False)
    preprocessing_time_ms = Column(Float, default=0.0)
    postprocessing_time_ms = Column(Float, default=0.0)

    # Visualization
    visualization_path = Column(String(500), nullable=True)

    # Extra data
    extra_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    inference_job = relationship("InferenceJob", back_populates="results")
    job = relationship("TrainingJob", foreign_keys=[training_job_id])
