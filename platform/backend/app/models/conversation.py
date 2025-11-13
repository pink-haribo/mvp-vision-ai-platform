"""
Conversation state machine and action schemas

This module defines:
- ConversationState: State machine states
- Action types: LLM response actions
- Pydantic schemas for validation
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ConversationState(str, Enum):
    """
    대화 상태 정의

    State transitions:
    initial → gathering_config → selecting_project → creating_project/confirming → complete
    initial → analyzing_dataset → selecting_model → gathering_config
    initial → running_inference
    initial → monitoring_training
    """

    INITIAL = "initial"
    """초기 상태 - 새로운 대화 시작"""

    GATHERING_CONFIG = "gathering_config"
    """학습 설정 수집 중 - 모델, 데이터셋, 하이퍼파라미터 등"""

    SELECTING_PROJECT = "selecting_project"
    """프로젝트 선택 중 - 신규/기존 프로젝트 선택"""

    CREATING_PROJECT = "creating_project"
    """프로젝트 생성 중 - 프로젝트 이름/설명 입력"""

    CONFIRMING = "confirming"
    """최종 확인 중 - 학습 시작 전 확인"""

    COMPLETE = "complete"
    """완료 - 학습 작업 생성 완료"""

    ERROR = "error"
    """오류 상태 - 복구 필요"""

    # Phase 1 추가 States
    ANALYZING_DATASET = "analyzing_dataset"
    """데이터셋 분석 중"""

    SELECTING_MODEL = "selecting_model"
    """모델 선택 중"""

    COMPARING_MODELS = "comparing_models"
    """모델 비교 중"""

    MONITORING_TRAINING = "monitoring_training"
    """학습 모니터링 중"""

    RUNNING_INFERENCE = "running_inference"
    """추론 실행 중"""

    VIEWING_RESULTS = "viewing_results"
    """결과 조회 중"""

    IDLE = "idle"
    """대기 중 - 다음 작업 대기"""


class ActionType(str, Enum):
    """
    LLM이 반환할 수 있는 액션 타입

    LLM은 사용자 메시지를 분석하고 적절한 action을 선택합니다.
    """

    ASK_CLARIFICATION = "ask_clarification"
    """추가 정보 필요 - 사용자에게 질문"""

    SHOW_PROJECT_OPTIONS = "show_project_options"
    """프로젝트 선택 옵션 표시 (1: 신규, 2: 기존, 3: 없이 진행)"""

    SHOW_PROJECT_LIST = "show_project_list"
    """기존 프로젝트 목록 표시"""

    CREATE_PROJECT = "create_project"
    """신규 프로젝트 생성"""

    SELECT_PROJECT = "select_project"
    """기존 프로젝트 선택"""

    SKIP_PROJECT = "skip_project"
    """프로젝트 없이 진행 (Uncategorized에 추가)"""

    CONFIRM_TRAINING = "confirm_training"
    """학습 시작 확인 요청"""

    START_TRAINING = "start_training"
    """학습 시작 (최종 완료)"""

    ERROR = "error"
    """오류 발생"""

    # Phase 1 추가 Actions - Dataset
    ANALYZE_DATASET = "analyze_dataset"
    """데이터셋 분석 실행"""

    SHOW_DATASET_ANALYSIS = "show_dataset_analysis"
    """데이터셋 분석 결과 표시"""

    LIST_DATASETS = "list_datasets"
    """사용 가능한 데이터셋 목록 표시"""

    # Phase 1 추가 Actions - Model
    SEARCH_MODELS = "search_models"
    """모델 검색"""

    SHOW_MODEL_INFO = "show_model_info"
    """모델 상세 정보 표시"""

    COMPARE_MODELS = "compare_models"
    """모델 비교"""

    RECOMMEND_MODELS = "recommend_models"
    """모델 추천"""

    # Phase 1 추가 Actions - Training Control
    STOP_TRAINING = "stop_training"
    """학습 중지"""

    RESUME_TRAINING = "resume_training"
    """학습 재개"""

    SHOW_TRAINING_STATUS = "show_training_status"
    """학습 상태 표시"""

    LIST_TRAINING_JOBS = "list_training_jobs"
    """학습 작업 목록 표시"""

    # Phase 1 추가 Actions - Inference
    START_QUICK_INFERENCE = "start_quick_inference"
    """빠른 추론 시작"""

    START_BATCH_INFERENCE = "start_batch_inference"
    """배치 추론 시작"""

    SHOW_INFERENCE_RESULTS = "show_inference_results"
    """추론 결과 표시"""

    # Phase 1 추가 Actions - Results
    SHOW_VALIDATION_RESULTS = "show_validation_results"
    """검증 결과 표시"""

    SHOW_CONFUSION_MATRIX = "show_confusion_matrix"
    """Confusion Matrix 표시"""

    # Phase 1 추가 Actions - General
    SHOW_HELP = "show_help"
    """도움말 표시"""

    RESET_CONVERSATION = "reset_conversation"
    """대화 초기화"""


# ========== Action Schemas ==========

class ActionBase(BaseModel):
    """Base action schema"""
    action: ActionType
    message: str = Field(..., description="사용자에게 보여줄 메시지")


class AskClarificationAction(ActionBase):
    """추가 정보 요청 액션"""
    action: ActionType = ActionType.ASK_CLARIFICATION
    missing_fields: List[str] = Field(default_factory=list, description="부족한 필드 목록")
    current_config: Dict[str, Any] = Field(default_factory=dict, description="현재까지 수집된 설정")


class ShowProjectOptionsAction(ActionBase):
    """프로젝트 선택 옵션 표시 액션"""
    action: ActionType = ActionType.SHOW_PROJECT_OPTIONS
    config: Dict[str, Any] = Field(..., description="완성된 학습 설정")
    experiment: Optional[Dict[str, Any]] = Field(None, description="실험 메타데이터")


class ShowProjectListAction(ActionBase):
    """프로젝트 목록 표시 액션"""
    action: ActionType = ActionType.SHOW_PROJECT_LIST


class CreateProjectAction(ActionBase):
    """프로젝트 생성 액션"""
    action: ActionType = ActionType.CREATE_PROJECT
    project_name: str = Field(..., description="프로젝트 이름")
    project_description: Optional[str] = Field(None, description="프로젝트 설명")
    task_type: Optional[str] = Field(None, description="작업 유형")


class SelectProjectAction(ActionBase):
    """프로젝트 선택 액션"""
    action: ActionType = ActionType.SELECT_PROJECT
    project_identifier: str = Field(..., description="프로젝트 ID 또는 이름")


class SkipProjectAction(ActionBase):
    """프로젝트 건너뛰기 액션"""
    action: ActionType = ActionType.SKIP_PROJECT


class ConfirmTrainingAction(ActionBase):
    """학습 확인 액션"""
    action: ActionType = ActionType.CONFIRM_TRAINING
    config: Dict[str, Any] = Field(..., description="학습 설정")
    project_id: Optional[int] = Field(None, description="선택된 프로젝트 ID")


class StartTrainingAction(ActionBase):
    """학습 시작 액션"""
    action: ActionType = ActionType.START_TRAINING
    config: Dict[str, Any] = Field(..., description="학습 설정")
    project_id: Optional[int] = Field(None, description="프로젝트 ID")
    experiment: Optional[Dict[str, Any]] = Field(None, description="실험 메타데이터")


class ErrorAction(ActionBase):
    """오류 액션"""
    action: ActionType = ActionType.ERROR
    error_message: str = Field(..., description="오류 메시지")


# ========== Training Config Schema ==========

class TrainingConfig(BaseModel):
    """학습 설정 스키마"""
    framework: Optional[str] = Field(None, description="프레임워크 (timm, ultralytics)")
    model_name: Optional[str] = Field(None, description="모델 이름")
    task_type: Optional[str] = Field(None, description="작업 유형")
    dataset_path: Optional[str] = Field(None, description="데이터셋 경로")
    dataset_format: Optional[str] = Field(None, description="데이터셋 포맷")
    num_classes: Optional[int] = Field(None, description="클래스 개수")
    epochs: Optional[int] = Field(None, description="에포크 수")
    batch_size: Optional[int] = Field(None, description="배치 크기")
    learning_rate: Optional[float] = Field(None, description="학습률")

    class Config:
        protected_namespaces = ()  # Allow model_name field

    def is_complete(self) -> bool:
        """모든 필수 필드가 채워졌는지 확인"""
        required_fields = [
            "framework",
            "model_name",
            "task_type",
            "dataset_path",
            "epochs",
            "batch_size",
            "learning_rate",
        ]
        return all(getattr(self, field) is not None for field in required_fields)

    def get_missing_fields(self) -> List[str]:
        """부족한 필드 목록 반환"""
        required_fields = [
            "framework",
            "model_name",
            "task_type",
            "dataset_path",
            "epochs",
            "batch_size",
            "learning_rate",
        ]
        return [field for field in required_fields if getattr(self, field) is None]


class ExperimentMetadata(BaseModel):
    """실험 메타데이터 스키마"""
    name: Optional[str] = Field(None, description="실험 이름")
    tags: List[str] = Field(default_factory=list, description="태그")
    notes: Optional[str] = Field(None, description="노트")


# ========== Gemini Response Schema (for structured output) ==========

class GeminiActionResponse(BaseModel):
    """
    Gemini structured output schema

    LLM이 반환하는 구조화된 응답
    """
    action: ActionType = Field(..., description="수행할 액션")
    message: str = Field(..., description="사용자에게 보여줄 메시지")

    # Optional fields based on action type
    missing_fields: Optional[List[str]] = Field(None, description="부족한 필드 (ask_clarification)")
    current_config: Optional[Dict[str, Any]] = Field(None, description="현재 설정 (ask_clarification)")

    config: Optional[Dict[str, Any]] = Field(None, description="완성된 설정 (show_project_options, start_training)")
    experiment: Optional[Dict[str, Any]] = Field(None, description="실험 메타데이터")

    project_name: Optional[str] = Field(None, description="프로젝트 이름 (create_project)")
    project_description: Optional[str] = Field(None, description="프로젝트 설명 (create_project)")
    project_identifier: Optional[str] = Field(None, description="프로젝트 식별자 (select_project)")

    project_id: Optional[int] = Field(None, description="프로젝트 ID (confirm_training, start_training)")

    error_message: Optional[str] = Field(None, description="오류 메시지 (error)")

    class Config:
        use_enum_values = True  # Enum을 문자열로 직렬화
