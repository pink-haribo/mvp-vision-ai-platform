"""
Action handlers for conversation actions

Each action type has a corresponding handler that executes the actual logic.
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models.conversation import (
    ActionType,
    GeminiActionResponse,
    ConversationState,
)
from app.db.models import (
    Session as SessionModel,
    Message as MessageModel,
    Project,
    TrainingJob,
)

logger = logging.getLogger(__name__)


class ActionHandlers:
    """
    Handles all conversation actions

    Each handler returns:
    - new_state: New conversation state
    - message: Message to show user
    - temp_data: Updated temporary data
    """

    def __init__(self, db: Session):
        self.db = db
        self.current_user_id = None  # Will be set during handle_action

    async def handle_action(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str,
        user_id: int  # Required: Current logged-in user ID
    ) -> Dict[str, Any]:
        """
        Route action to appropriate handler

        Args:
            action_response: LLM's action response
            session: Current session
            user_message: Original user message
            user_id: Current user ID for project ownership - REQUIRED

        Returns:
            dict: {
                "new_state": ConversationState,
                "message": str,
                "temp_data": dict,
                "training_job_id": int (optional)
            }
        """
        # Store current user ID for handlers to use
        self.current_user_id = user_id

        # CRITICAL: Apply fallback extraction BEFORE routing to handler
        # This ensures config data is extracted even if LLM fails
        temp_data = session.temp_data or {}
        existing_config = temp_data.get("config", {})

        # TRACE: Step 4 - Before merging
        print(f"\n[TRACE-4-MERGE] Action handler:")
        print(f"  existing_config (from session): {existing_config}")
        print(f"  action_response.current_config: {action_response.current_config}")

        # Merge LLM's config first
        if action_response.current_config:
            existing_config.update(action_response.current_config)
            print(f"  MERGED config: {existing_config}")
        else:
            print(f"  NO MERGE - action_response.current_config is None/empty")

        # Then apply fallback extraction from user message
        # CRITICAL DEBUG: Write to file
        try:
            import os
            import datetime
            log_path = "C:\\Users\\flyto\\Project\\Github\\mvp-vision-ai-platform\\mvp\\data\\logs\\fallback_debug.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            with open(log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n[{timestamp}] Action: {action_response.action}\n")
                f.write(f"Before: {existing_config}\n")
                f.write(f"User message: {user_message}\n")
        except Exception as e:
            print(f"LOG ERROR: {e}")

        existing_config = self._extract_from_user_message(user_message, existing_config)

        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"After: {existing_config}\n")
        except:
            pass

        logger.warning(f"[DEBUG] Before extraction: {existing_config}")
        logger.warning(f"[DEBUG] After extraction: {existing_config}")

        # Update session temp_data with extracted config
        temp_data["config"] = existing_config
        session.temp_data = temp_data

        logger.warning(f"[FALLBACK] Config after extraction: {existing_config}")

        action = action_response.action

        handlers = {
            # 기존 핸들러
            ActionType.ASK_CLARIFICATION: self._handle_ask_clarification,
            ActionType.SHOW_PROJECT_OPTIONS: self._handle_show_project_options,
            ActionType.SHOW_PROJECT_LIST: self._handle_show_project_list,
            ActionType.CREATE_PROJECT: self._handle_create_project,
            ActionType.SELECT_PROJECT: self._handle_select_project,
            ActionType.SKIP_PROJECT: self._handle_skip_project,
            ActionType.CONFIRM_TRAINING: self._handle_confirm_training,
            ActionType.START_TRAINING: self._handle_start_training,
            ActionType.ERROR: self._handle_error,

            # Phase 1 추가 핸들러 - Dataset
            ActionType.ANALYZE_DATASET: self._handle_analyze_dataset,
            ActionType.SHOW_DATASET_ANALYSIS: self._handle_show_dataset_analysis,
            ActionType.LIST_DATASETS: self._handle_list_datasets,

            # Phase 1 추가 핸들러 - Model
            ActionType.SEARCH_MODELS: self._handle_search_models,
            ActionType.SHOW_MODEL_INFO: self._handle_show_model_info,
            ActionType.RECOMMEND_MODELS: self._handle_recommend_models,
            ActionType.COMPARE_MODELS: self._handle_compare_models,

            # Phase 1 추가 핸들러 - Training Control
            ActionType.SHOW_TRAINING_STATUS: self._handle_show_training_status,
            ActionType.STOP_TRAINING: self._handle_stop_training,
            ActionType.LIST_TRAINING_JOBS: self._handle_list_training_jobs,
            ActionType.RESUME_TRAINING: self._handle_resume_training,

            # Phase 1 추가 핸들러 - Inference
            ActionType.START_QUICK_INFERENCE: self._handle_start_quick_inference,
            ActionType.START_BATCH_INFERENCE: self._handle_start_batch_inference,
            ActionType.SHOW_INFERENCE_RESULTS: self._handle_show_inference_results,

            # Phase 1 추가 핸들러 - Results
            ActionType.SHOW_VALIDATION_RESULTS: self._handle_show_validation_results,
            ActionType.SHOW_CONFUSION_MATRIX: self._handle_show_confusion_matrix,

            # Phase 1 추가 핸들러 - Utility
            ActionType.SHOW_HELP: self._handle_show_help,
            ActionType.RESET_CONVERSATION: self._handle_reset_conversation,
        }

        handler = handlers.get(action)
        if not handler:
            logger.error(f"Unknown action: {action}")
            return self._handle_error(action_response, session, user_message)

        # Call handler
        result = await handler(action_response, session, user_message)

        # CRITICAL: Merge our extracted config with handler's temp_data
        # This ensures extracted data isn't lost when handler returns
        # EXCEPTION: Skip merge for actions that intentionally clear state
        if action == ActionType.RESET_CONVERSATION:
            # Don't merge config back - handler wants to clear everything
            logger.info(f"[RESET] Skipping config merge for {action}")
            return result

        handler_temp_data = result.get("temp_data", {})
        handler_config = handler_temp_data.get("config", {})

        # Merge: extracted config (priority) + handler config
        final_config = {**handler_config, **existing_config}

        handler_temp_data["config"] = final_config
        result["temp_data"] = handler_temp_data

        logger.info(f"[MERGE] Final config: {final_config}")

        return result

    async def _handle_ask_clarification(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle ask_clarification action"""
        temp_data = session.temp_data or {}
        current_state = ConversationState(session.state)

        # Config is already extracted in handle_action, just retrieve it
        existing_config = temp_data.get("config", {})

        # Determine next state based on missing fields
        missing_fields = action_response.missing_fields or []

        # If asking for project_name, transition to CREATING_PROJECT
        if "project_name" in missing_fields:
            new_state = ConversationState.CREATING_PROJECT
        # If asking for config fields, stay in or go to GATHERING_CONFIG
        else:
            new_state = ConversationState.GATHERING_CONFIG

        logger.debug(f"After ask_clarification: config = {existing_config}")

        return {
            "new_state": new_state,
            "message": action_response.message,
            "temp_data": temp_data,
        }

    async def _handle_show_project_options(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_project_options action"""
        temp_data = session.temp_data or {}

        # Config is already extracted in handle_action, just retrieve it
        existing_config = temp_data.get("config", {})

        # Save experiment metadata
        if action_response.experiment:
            temp_data["experiment"] = action_response.experiment

        logger.debug(f"After show_project_options: config = {existing_config}")

        # Build project options message with config summary
        message = self._format_config_summary(existing_config)
        message += "\n\n학습 설정이 완료되었습니다. 프로젝트를 선택해주세요.\n\n"
        message += "1️⃣ 신규 프로젝트 생성\n"
        message += "2️⃣ 기존 프로젝트 선택\n"
        message += "3️⃣ 프로젝트 없이 실험만 진행\n\n"
        message += "원하시는 방식의 번호를 입력해주세요."

        return {
            "new_state": ConversationState.SELECTING_PROJECT,
            "message": message,
            "temp_data": temp_data,
        }

    async def _handle_show_project_list(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_project_list action"""
        # Fetch projects (excluding Uncategorized)
        projects = self.db.query(Project).filter(
            Project.name != "Uncategorized"
        ).order_by(Project.updated_at.desc()).all()

        temp_data = session.temp_data or {}

        if not projects:
            message = "사용 가능한 프로젝트가 없습니다.\n\n"
            message += "다른 옵션을 선택하시겠습니까?\n"
            message += "1️⃣ 신규 프로젝트 생성\n"
            message += "3️⃣ 프로젝트 없이 실험만 진행"

            return {
                "new_state": ConversationState.SELECTING_PROJECT,
                "message": message,
                "temp_data": temp_data,
            }

        # Build project list
        message = "다음 프로젝트 중 하나를 선택해주세요:\n\n"
        available_projects = []

        for idx, project in enumerate(projects, start=1):
            desc = f" - {project.description}" if project.description else ""
            task = f" ({project.task_type})" if project.task_type else ""

            # Count experiments
            exp_count = self.db.query(TrainingJob).filter(
                TrainingJob.project_id == project.id
            ).count()

            message += f"{idx}. **{project.name}**{task}{desc} (실험 {exp_count}개)\n"

            available_projects.append({
                "id": project.id,
                "name": project.name,
            })

        message += "\n프로젝트 번호를 입력하거나 프로젝트 이름을 입력해주세요."

        # Save available projects to temp_data
        temp_data["available_projects"] = available_projects

        return {
            "new_state": ConversationState.SELECTING_PROJECT,
            "message": message,
            "temp_data": temp_data,
        }

    async def _handle_create_project(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle create_project action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # DEBUG: Log what we received
        logger.info(f"[CREATE_PROJECT] action_response.project_name = {action_response.project_name}")
        logger.info(f"[CREATE_PROJECT] action_response.project_description = {action_response.project_description}")
        logger.info(f"[CREATE_PROJECT] Full action_response: {action_response}")

        # Validate project name (required)
        project_name = action_response.project_name
        if not project_name or str(project_name).strip() == "":
            logger.warning("create_project called without project_name")
            return {
                "new_state": ConversationState.SELECTING_PROJECT,
                "message": "프로젝트 이름을 입력해주세요.\n\n예시: 이미지 분류 프로젝트",
                "temp_data": temp_data
            }

        # Create new project
        try:
            new_project = Project(
                name=project_name.strip(),
                description=action_response.project_description or "",
                task_type=config.get("task_type"),
                user_id=self.current_user_id  # Set owner to current logged-in user
            )
            self.db.add(new_project)
            self.db.commit()
            self.db.refresh(new_project)

            logger.info(f"Created project: {new_project.name} (ID: {new_project.id})")
        except Exception as e:
            logger.error(f"Failed to create project: {e}", exc_info=True)
            self.db.rollback()
            return {
                "new_state": ConversationState.ERROR,
                "message": f"프로젝트 생성 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

        # Save project ID to temp_data
        temp_data["selected_project_id"] = new_project.id

        # Build confirmation message
        message = f"프로젝트 '{new_project.name}'이(가) 생성되었습니다.\n\n"
        message += self._format_config_summary(config)
        message += "\n\n학습을 시작하시겠습니까? (예/아니오)"

        return {
            "new_state": ConversationState.CONFIRMING,
            "message": message,
            "temp_data": temp_data,
        }

    async def _handle_select_project(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle select_project action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        project_identifier = action_response.project_identifier

        # Try to find project
        project = None

        # Check if identifier is a number (project index)
        if project_identifier.isdigit():
            available_projects = temp_data.get("available_projects", [])
            project_idx = int(project_identifier) - 1

            if 0 <= project_idx < len(available_projects):
                project_id = available_projects[project_idx]["id"]
                project = self.db.query(Project).filter(Project.id == project_id).first()

        # If not found, try to search by name
        if not project:
            project = self.db.query(Project).filter(
                Project.name.ilike(f"%{project_identifier}%")
            ).first()

        if not project:
            return {
                "new_state": ConversationState.SELECTING_PROJECT,
                "message": f"'{project_identifier}' 프로젝트를 찾을 수 없습니다. 다시 선택해주세요.",
                "temp_data": temp_data,
            }

        # Save selected project
        temp_data["selected_project_id"] = project.id

        # Build confirmation message
        message = f"프로젝트 '{project.name}'을(를) 선택했습니다.\n\n"
        message += self._format_config_summary(config)
        message += "\n\n학습을 시작하시겠습니까? (예/아니오)"

        return {
            "new_state": ConversationState.CONFIRMING,
            "message": message,
            "temp_data": temp_data,
            "selected_project_id": project.id,  # For frontend to show project detail
        }

    async def _handle_skip_project(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle skip_project action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # Get or create Uncategorized project
        uncategorized = self.db.query(Project).filter(
            Project.name == "Uncategorized"
        ).first()

        if not uncategorized:
            uncategorized = Project(
                name="Uncategorized",
                description="프로젝트 없이 진행한 실험들",
            )
            self.db.add(uncategorized)
            self.db.commit()
            self.db.refresh(uncategorized)

        temp_data["selected_project_id"] = uncategorized.id

        # Build confirmation message
        message = "프로젝트 없이 진행합니다.\n\n"
        message += self._format_config_summary(config)
        message += "\n\n학습을 시작하시겠습니까? (예/아니오)"

        return {
            "new_state": ConversationState.CONFIRMING,
            "message": message,
            "temp_data": temp_data,
        }

    async def _handle_confirm_training(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle confirm_training action"""
        # This is just a confirmation display, wait for user response
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        message = self._format_config_summary(config)
        message += "\n\n학습을 시작하시겠습니까? (예/아니오)"

        return {
            "new_state": ConversationState.CONFIRMING,
            "message": message,
            "temp_data": temp_data,
        }

    async def _handle_start_training(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle start_training action"""
        from app.db.models import Dataset

        temp_data = session.temp_data or {}
        config = action_response.config or temp_data.get("config", {})
        experiment = action_response.experiment or temp_data.get("experiment", {})
        project_id = action_response.project_id or temp_data.get("selected_project_id")

        # Resolve dataset: prefer dataset_id over dataset_path
        dataset_id = config.get("dataset_id")
        dataset_path = config.get("dataset_path")
        dataset_format = config.get("dataset_format", "imagefolder")

        if dataset_id:
            # Look up dataset in DB
            dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                # Use dataset info from DB
                dataset_path = dataset_id  # Training Service will resolve this
                dataset_format = dataset.format
                logger.info(f"[START_TRAINING] Using dataset from DB: {dataset_id}")
            else:
                logger.warning(f"[START_TRAINING] Dataset ID '{dataset_id}' not found in DB, using path")

        # Create training job
        training_job = TrainingJob(
            session_id=session.id,
            project_id=project_id,
            framework=config.get("framework"),
            model_name=config.get("model_name"),
            task_type=config.get("task_type"),
            dataset_id=dataset_id,  # Store dataset ID
            dataset_path=dataset_path,  # Use resolved path or ID
            dataset_format=dataset_format,  # Use format from DB or config
            num_classes=config.get("num_classes"),
            epochs=config.get("epochs"),
            batch_size=config.get("batch_size"),
            learning_rate=config.get("learning_rate"),
            output_dir=f"./outputs/{session.id}",
            experiment_name=experiment.get("name"),
            tags=experiment.get("tags"),
            notes=experiment.get("notes"),
            status="pending",
        )
        self.db.add(training_job)
        self.db.commit()
        self.db.refresh(training_job)

        logger.info(f"Created training job: ID={training_job.id}, dataset_id={dataset_id}, dataset_path={dataset_path}")

        message = f"학습 작업이 생성되었습니다! (Job ID: {training_job.id})\n\n"
        message += "학습이 시작됩니다. 우측 패널에서 진행 상황을 확인하실 수 있습니다."

        return {
            "new_state": ConversationState.COMPLETE,
            "message": message,
            "temp_data": {},  # Clear temp data
            "training_job_id": training_job.id,
        }

    async def _handle_error(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle error action"""
        error_msg = action_response.error_message or "알 수 없는 오류가 발생했습니다."
        logger.error(f"Action error: {error_msg}")

        return {
            "new_state": ConversationState.ERROR,
            "message": f"죄송합니다. 오류가 발생했습니다: {error_msg}\n\n처음부터 다시 시작해주세요.",
            "temp_data": {},
        }

    def _extract_from_user_message(
        self, user_message: str, existing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract config values from user message (fallback for LLM limitations)

        This handles cases where LLM doesn't properly extract structured data.
        """
        import re
        import os

        msg_lower = user_message.lower().strip()

        # Extract dataset path (Windows/Unix paths)
        # Match patterns like: C:\path\to\dataset or /path/to/dataset
        path_pattern = r'[A-Za-z]:\\[\w\\\-\.]+|/[\w/\-\.]+'
        path_matches = re.findall(path_pattern, user_message)
        if path_matches:
            # Take the longest match (most likely to be the full path)
            dataset_path = max(path_matches, key=len)
            if 'dataset' in dataset_path.lower() or os.path.exists(dataset_path):
                existing_config["dataset_path"] = dataset_path
                logger.info(f"Extracted dataset_path from user message: {dataset_path}")

        # Extract default values (Korean & English)
        if any(keyword in msg_lower for keyword in ["기본", "default", "기본값"]):
            if "epochs" not in existing_config or existing_config.get("epochs") is None:
                existing_config["epochs"] = 50
                logger.info("Applied default epochs: 50")
            if "batch_size" not in existing_config or existing_config.get("batch_size") is None:
                existing_config["batch_size"] = 32
                logger.info("Applied default batch_size: 32")
            if "learning_rate" not in existing_config or existing_config.get("learning_rate") is None:
                existing_config["learning_rate"] = 0.001
                logger.info("Applied default learning_rate: 0.001")
            if "dataset_format" not in existing_config or existing_config.get("dataset_format") is None:
                existing_config["dataset_format"] = "imagefolder"

        # Extract epochs (숫자 + "epoch" or "에포크")
        epoch_match = re.search(r'(\d+)\s*(?:epoch|에포크)', msg_lower)
        if epoch_match:
            existing_config["epochs"] = int(epoch_match.group(1))
            logger.info(f"Extracted epochs: {existing_config['epochs']}")

        # Extract batch size (숫자 + "batch" or "배치")
        batch_match = re.search(r'(?:batch|배치)[\s:]*(\d+)', msg_lower)
        if batch_match:
            existing_config["batch_size"] = int(batch_match.group(1))
            logger.info(f"Extracted batch_size: {existing_config['batch_size']}")

        # Extract learning rate
        lr_match = re.search(r'(?:lr|learning.?rate|학습률)[\s:=]*(0?\.\d+)', msg_lower)
        if lr_match:
            existing_config["learning_rate"] = float(lr_match.group(1))
            logger.info(f"Extracted learning_rate: {existing_config['learning_rate']}")

        return existing_config

    def _format_config_summary(self, config: Dict[str, Any]) -> str:
        """Format config as readable summary"""
        lines = [
            "**학습 설정:**",
            f"- 프레임워크: {config.get('framework', 'N/A')}",
            f"- 모델: {config.get('model_name', 'N/A')}",
            f"- 작업 유형: {config.get('task_type', 'N/A')}",
            f"- 데이터셋: {config.get('dataset_path', 'N/A')}",
            f"- 에포크: {config.get('epochs', 'N/A')}",
            f"- 배치 크기: {config.get('batch_size', 'N/A')}",
            f"- 학습률: {config.get('learning_rate', 'N/A')}",
        ]

        # Add advanced config preset if specified
        advanced_config = config.get('advanced_config')
        if advanced_config:
            preset_descriptions = {
                "basic": "간단한 학습 (minimal augmentation, Adam optimizer)",
                "standard": "균형잡힌 설정 (AdamW, cosine scheduler, moderate augmentation)",
                "aggressive": "강력한 augmentation (작은 데이터셋에 적합)",
                "fine_tuning": "사전학습 모델 fine-tuning 최적화"
            }
            preset_desc = preset_descriptions.get(advanced_config, advanced_config)
            lines.append(f"- Advanced Config: {advanced_config} ({preset_desc})")

        return "\n".join(lines)

    # ========== Phase 1 Dataset Handlers ==========

    async def _handle_analyze_dataset(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle analyze_dataset action

        Analyzes a dataset's structure, format, and quality using Tool Registry.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # Get dataset path from config or action response
        dataset_path = config.get("dataset_path")

        if not dataset_path:
            logger.warning("analyze_dataset called without dataset_path")
            return {
                "new_state": ConversationState.INITIAL,
                "message": "데이터셋 경로를 알려주세요. 예: C:/datasets/my_dataset",
                "temp_data": temp_data
            }

        # Call tool registry to analyze dataset
        try:
            logger.info(f"Analyzing dataset at: {dataset_path}")
            result = await tool_registry.call_tool(
                "analyze_dataset",
                {"dataset_path": dataset_path},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            # Save analysis results to temp_data
            temp_data["dataset_analysis"] = result

            # Format analysis results for user
            message = self._format_dataset_analysis(result)

            return {
                "new_state": ConversationState.ANALYZING_DATASET,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to analyze dataset: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"데이터셋 분석 중 오류가 발생했습니다: {str(e)}\n\n경로를 확인하고 다시 시도해주세요.",
                "temp_data": temp_data
            }

    async def _handle_show_dataset_analysis(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle show_dataset_analysis action

        Displays previously analyzed dataset information from temp_data.
        """
        temp_data = session.temp_data or {}
        analysis = temp_data.get("dataset_analysis")

        if not analysis:
            logger.warning("show_dataset_analysis called without prior analysis")
            return {
                "new_state": ConversationState.INITIAL,
                "message": "먼저 데이터셋을 분석해주세요. 데이터셋 경로를 알려주시면 분석해드리겠습니다.",
                "temp_data": temp_data
            }

        # Format and display analysis
        message = self._format_dataset_analysis(analysis)
        message += "\n\n이 데이터셋으로 학습을 진행하시겠습니까?"

        return {
            "new_state": ConversationState.ANALYZING_DATASET,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_list_datasets(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle list_datasets action

        Lists available datasets in default or specified directory.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}

        # Get base path from action response or use default
        base_path = "C:/datasets"  # Default path

        try:
            logger.info(f"Listing datasets in: {base_path}")
            datasets = await tool_registry.call_tool(
                "list_datasets",
                {"base_path": base_path},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            if not datasets:
                message = f"{base_path}에 사용 가능한 데이터셋이 없습니다.\n\n"
                message += "데이터셋 경로를 직접 입력해주세요."

                return {
                    "new_state": ConversationState.INITIAL,
                    "message": message,
                    "temp_data": temp_data
                }

            # Format dataset list
            message = f"**사용 가능한 데이터셋** ({base_path}):\n\n"
            for idx, dataset in enumerate(datasets, start=1):
                message += f"{idx}. {dataset['name']}\n"
                message += f"   경로: {dataset['path']}\n\n"

            message += "사용할 데이터셋 이름 또는 경로를 입력해주세요."

            # Save dataset list to temp_data for reference
            temp_data["available_datasets"] = datasets

            return {
                "new_state": ConversationState.INITIAL,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.INITIAL,
                "message": f"데이터셋 목록 조회 중 오류가 발생했습니다: {str(e)}\n\n데이터셋 경로를 직접 입력해주세요.",
                "temp_data": temp_data
            }

    def _format_dataset_analysis(self, analysis: Dict[str, Any]) -> str:
        """
        Format dataset analysis results for user display

        Args:
            analysis: Analysis results from tool_registry.analyze_dataset

        Returns:
            Formatted message string
        """
        lines = ["**데이터셋 분석 결과:**\n"]

        # Basic info
        lines.append(f"📁 경로: {analysis.get('path', 'N/A')}")
        lines.append(f"📋 포맷: {analysis.get('format', 'unknown')}")
        lines.append(f"📊 총 이미지 수: {analysis.get('total_images', 0):,}개")
        lines.append(f"🏷️ 클래스 수: {analysis.get('num_classes', 0)}개")

        # Class distribution
        classes = analysis.get('classes', [])
        if classes:
            lines.append(f"\n**클래스 목록:**")
            class_dist = analysis.get('class_distribution', {})
            for cls in classes[:10]:  # Show first 10 classes
                count = class_dist.get(cls, 0)
                lines.append(f"  - {cls}: {count:,}개")

            if len(classes) > 10:
                lines.append(f"  ... 외 {len(classes) - 10}개 클래스")

        # Dataset info/warnings
        dataset_info = analysis.get('dataset_info', {})
        if dataset_info:
            lines.append(f"\n**데이터셋 정보:**")
            for key, value in dataset_info.items():
                lines.append(f"  - {key}: {value}")

        # Suggestions
        suggestions = analysis.get('suggestions', [])
        if suggestions:
            lines.append(f"\n**💡 권장사항:**")
            for suggestion in suggestions:
                lines.append(f"  - {suggestion}")

        return "\n".join(lines)

    # ========== Phase 1 Model Handlers ==========

    async def _handle_search_models(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle search_models action

        Searches for models based on task type, framework, or tags.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # Extract search parameters from config
        search_params = {}
        if config.get("task_type"):
            search_params["task_type"] = config["task_type"]
        if config.get("framework"):
            search_params["framework"] = config["framework"]

        try:
            logger.info(f"Searching models with params: {search_params}")
            models = await tool_registry.call_tool(
                "search_models",
                search_params,
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            if not models:
                message = "검색 조건에 맞는 모델을 찾을 수 없습니다.\n\n"
                message += "다른 조건으로 다시 검색해주세요."

                return {
                    "new_state": ConversationState.SELECTING_MODEL,
                    "message": message,
                    "temp_data": temp_data
                }

            # Save search results to temp_data
            temp_data["model_search_results"] = models

            # Format model list
            message = self._format_model_list(models, search_params)
            message += "\n\n사용할 모델을 선택해주세요."

            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to search models: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"모델 검색 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_show_model_info(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle show_model_info action

        Shows detailed information about a specific model.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # Get model info from config
        framework = config.get("framework")
        model_name = config.get("model_name")

        if not framework or not model_name:
            logger.warning("show_model_info called without framework/model_name")
            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": "모델 정보를 확인하려면 프레임워크와 모델 이름이 필요합니다.\n\n예: timm의 resnet50 정보를 알려줘",
                "temp_data": temp_data
            }

        try:
            logger.info(f"Getting model guide for: {framework}/{model_name}")
            model_guide = await tool_registry.call_tool(
                "get_model_guide",
                {"framework": framework, "model_name": model_name},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            # Format model info
            message = self._format_model_info(model_guide)
            message += "\n\n이 모델로 학습을 진행하시겠습니까?"

            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"모델 정보 조회 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_recommend_models(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle recommend_models action

        Recommends models based on dataset analysis and task type.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})
        dataset_analysis = temp_data.get("dataset_analysis", {})

        # Determine task type from config or dataset
        task_type = config.get("task_type")

        # If no task type specified, try to infer from dataset
        if not task_type:
            # Default to classification if we have class information
            if dataset_analysis.get("num_classes", 0) > 0:
                task_type = "classification"
                config["task_type"] = task_type

        if not task_type:
            logger.warning("recommend_models called without task_type")
            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": "모델을 추천하려면 작업 유형이 필요합니다.\n\n어떤 작업을 하시겠습니까? (예: 분류, 객체 검출, 세그멘테이션)",
                "temp_data": temp_data
            }

        try:
            # Search models for the task type
            logger.info(f"Recommending models for task: {task_type}")
            models = await tool_registry.call_tool(
                "search_models",
                {"task_type": task_type},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            if not models:
                message = f"{task_type} 작업에 적합한 모델을 찾을 수 없습니다."
                return {
                    "new_state": ConversationState.SELECTING_MODEL,
                    "message": message,
                    "temp_data": temp_data
                }

            # Sort by recommendation (for now, just take first 3)
            recommended = models[:3]
            temp_data["recommended_models"] = recommended

            # Format recommendations
            message = f"**{task_type} 작업에 추천하는 모델:**\n\n"

            num_classes = dataset_analysis.get("num_classes", 0)
            if num_classes > 0:
                message += f"데이터셋 분석 결과 {num_classes}개 클래스가 발견되었습니다.\n\n"

            for idx, model in enumerate(recommended, start=1):
                message += f"{idx}. **{model['name']}** ({model['framework']})\n"
                message += f"   {model.get('description', 'No description')}\n\n"

            message += "사용할 모델 번호를 선택하거나 모델 이름을 입력해주세요."

            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to recommend models: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"모델 추천 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    def _format_model_list(
        self, models: list, search_params: Dict[str, Any]
    ) -> str:
        """
        Format model search results for user display

        Args:
            models: List of model dictionaries
            search_params: Search parameters used

        Returns:
            Formatted message string
        """
        lines = ["**모델 검색 결과:**\n"]

        # Show search criteria
        if search_params:
            lines.append("검색 조건:")
            for key, value in search_params.items():
                lines.append(f"  - {key}: {value}")
            lines.append("")

        # List models
        lines.append(f"총 {len(models)}개의 모델을 찾았습니다:\n")

        for idx, model in enumerate(models, start=1):
            lines.append(f"{idx}. **{model['name']}** ({model['framework']})")
            lines.append(f"   작업 유형: {', '.join(model.get('task_types', []))}")
            if model.get('description'):
                lines.append(f"   설명: {model['description']}")
            lines.append("")

        return "\n".join(lines)

    def _format_model_info(self, model_guide: Dict[str, Any]) -> str:
        """
        Format model guide information for user display

        Args:
            model_guide: Model guide from tool_registry

        Returns:
            Formatted message string
        """
        lines = ["**모델 상세 정보:**\n"]

        lines.append(f"📦 프레임워크: {model_guide.get('framework', 'N/A')}")
        lines.append(f"🏷️ 모델명: {model_guide.get('model_name', 'N/A')}")
        lines.append(f"📝 설명: {model_guide.get('description', 'N/A')}")
        lines.append(f"✅ 사용 가능: {'예' if model_guide.get('available') else '아니오'}")

        # Additional details if available
        if model_guide.get('parameters'):
            lines.append(f"\n**파라미터:**")
            for key, value in model_guide['parameters'].items():
                lines.append(f"  - {key}: {value}")

        if model_guide.get('performance'):
            lines.append(f"\n**성능:**")
            for key, value in model_guide['performance'].items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    # ========== Phase 1 Training Control Handlers ==========

    async def _handle_show_training_status(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle show_training_status action

        Shows current status and progress of a training job.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}

        # Try to get job_id from session's most recent training job
        job_id = None

        # Check if user specified a job_id in the message
        import re
        job_match = re.search(r'(?:job|작업)[\s#]*(\d+)', user_message.lower())
        if job_match:
            job_id = int(job_match.group(1))
        else:
            # Get most recent training job from this session
            recent_job = self.db.query(TrainingJob).filter(
                TrainingJob.session_id == session.id
            ).order_by(TrainingJob.created_at.desc()).first()

            if recent_job:
                job_id = recent_job.id

        if not job_id:
            logger.warning("show_training_status called without job_id")
            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": "학습 작업 ID를 알려주세요. 예: job 123의 상태를 알려줘",
                "temp_data": temp_data
            }

        try:
            logger.info(f"Getting training status for job: {job_id}")
            status = await tool_registry.call_tool(
                "get_training_status",
                {"job_id": job_id},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            # Format training status
            message = self._format_training_status(status)

            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to get training status: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 상태 조회 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_stop_training(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle stop_training action

        Stops a running training job.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}

        # Try to get job_id from user message
        import re
        job_id = None
        job_match = re.search(r'(?:job|작업)[\s#]*(\d+)', user_message.lower())
        if job_match:
            job_id = int(job_match.group(1))
        else:
            # Get most recent running job from this session
            running_job = self.db.query(TrainingJob).filter(
                TrainingJob.session_id == session.id,
                TrainingJob.status == "running"
            ).order_by(TrainingJob.created_at.desc()).first()

            if running_job:
                job_id = running_job.id

        if not job_id:
            logger.warning("stop_training called without job_id")
            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": "중지할 학습 작업 ID를 알려주세요. 예: job 123 중지해줘",
                "temp_data": temp_data
            }

        try:
            logger.info(f"Stopping training job: {job_id}")
            result = await tool_registry.call_tool(
                "stop_training",
                {"job_id": job_id, "save_checkpoint": True},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            message = f"**학습 중지 결과:**\n\n"
            message += f"Job ID: {result.get('job_id')}\n"
            message += f"상태: {result.get('status')}\n"
            message += f"{result.get('message')}"

            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to stop training: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 중지 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_list_training_jobs(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle list_training_jobs action

        Lists training jobs with optional filters.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        # Extract filters from user message
        filters = {"limit": 10}

        # Check for status filter
        if "실행중" in user_message or "running" in user_message.lower():
            filters["status"] = "running"
        elif "완료" in user_message or "complete" in user_message.lower():
            filters["status"] = "completed"
        elif "실패" in user_message or "failed" in user_message.lower():
            filters["status"] = "failed"

        try:
            logger.info(f"Listing training jobs with filters: {filters}")
            jobs = await tool_registry.call_tool(
                "list_training_jobs",
                filters,
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            if not jobs:
                message = "조건에 맞는 학습 작업이 없습니다."
                return {
                    "new_state": ConversationState.MONITORING_TRAINING,
                    "message": message,
                    "temp_data": temp_data
                }

            # Format job list
            message = f"**학습 작업 목록** (최근 {len(jobs)}개):\n\n"

            for job in jobs:
                message += f"📊 Job #{job['job_id']} - {job['model']}\n"
                message += f"   상태: {job['status']}\n"
                message += f"   작업 유형: {job.get('task_type', 'N/A')}\n"
                if job.get('final_metric'):
                    message += f"   최종 정확도: {job['final_metric']:.2%}\n"
                message += f"   생성: {job.get('created_at', 'N/A')}\n\n"

            message += "상세 정보를 확인하려면 'job X 상태 알려줘'라고 입력하세요."

            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to list training jobs: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 작업 목록 조회 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    # ========== Phase 1 Inference Handlers ==========

    async def _handle_start_quick_inference(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Handle start_quick_inference action

        Runs quick inference on a single image.
        """
        from app.utils.tool_registry import tool_registry

        temp_data = session.temp_data or {}

        # Try to extract job_id and image_path from message
        import re
        job_id = None
        image_path = None

        # Extract job_id
        job_match = re.search(r'(?:job|작업)[\s#]*(\d+)', user_message.lower())
        if job_match:
            job_id = int(job_match.group(1))
        else:
            # Get most recent completed job
            completed_job = self.db.query(TrainingJob).filter(
                TrainingJob.session_id == session.id,
                TrainingJob.status.in_(["completed", "running"])
            ).order_by(TrainingJob.created_at.desc()).first()

            if completed_job:
                job_id = completed_job.id

        # Extract image path
        path_pattern = r'[A-Za-z]:\\[\w\\\-\.]+\.(jpg|jpeg|png|bmp)|/[\w/\-\.]+\.(jpg|jpeg|png|bmp)'
        path_match = re.search(path_pattern, user_message, re.IGNORECASE)
        if path_match:
            image_path = path_match.group(0)

        if not job_id:
            logger.warning("start_quick_inference called without job_id")
            return {
                "new_state": ConversationState.RUNNING_INFERENCE,
                "message": "추론을 실행할 학습 작업 ID를 알려주세요. 예: job 123으로 이미지 추론해줘",
                "temp_data": temp_data
            }

        if not image_path:
            logger.warning("start_quick_inference called without image_path")
            return {
                "new_state": ConversationState.RUNNING_INFERENCE,
                "message": "추론할 이미지 경로를 알려주세요. 예: C:/images/test.jpg",
                "temp_data": temp_data
            }

        try:
            logger.info(f"Running inference: job={job_id}, image={image_path}")
            result = await tool_registry.call_tool(
                "run_quick_inference",
                {"job_id": job_id, "image_path": image_path},
                self.db,
                user_id=None  # Phase 1: Skip auth
            )

            # Format inference results
            message = f"**추론 결과:**\n\n"
            message += f"Job ID: {result.get('job_id')}\n"
            message += f"이미지: {result.get('image_path')}\n\n"

            predictions = result.get('predictions', [])
            if predictions:
                message += "예측:\n"
                for pred in predictions[:5]:  # Top 5 predictions
                    message += f"  - {pred.get('class')}: {pred.get('confidence', 0):.2%}\n"
            else:
                message += result.get('message', '추론이 완료되었습니다.')

            return {
                "new_state": ConversationState.RUNNING_INFERENCE,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to run inference: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"추론 실행 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    def _format_training_status(self, status: Dict[str, Any]) -> str:
        """
        Format training status for user display

        Args:
            status: Training status from tool_registry

        Returns:
            Formatted message string
        """
        lines = ["**학습 상태:**\n"]

        lines.append(f"📊 Job ID: {status.get('job_id')}")
        lines.append(f"🔧 모델: {status.get('model')}")
        lines.append(f"📦 프레임워크: {status.get('framework')}")
        lines.append(f"📈 상태: {status.get('status')}")

        # Progress
        current_epoch = status.get('current_epoch', 0)
        total_epochs = status.get('total_epochs', 0)
        progress = status.get('progress_percent', 0)

        lines.append(f"⏱️ 진행: {current_epoch}/{total_epochs} epochs ({progress:.1f}%)")

        # Latest metrics
        latest = status.get('latest_metrics', {})
        if latest:
            lines.append(f"\n**최근 메트릭 (Epoch {latest.get('epoch', 0)}):**")
            if latest.get('loss') is not None:
                lines.append(f"  - Loss: {latest['loss']:.4f}")
            if latest.get('accuracy') is not None:
                lines.append(f"  - Accuracy: {latest['accuracy']:.2%}")
            if latest.get('val_loss') is not None:
                lines.append(f"  - Val Loss: {latest['val_loss']:.4f}")
            if latest.get('val_accuracy') is not None:
                lines.append(f"  - Val Accuracy: {latest['val_accuracy']:.2%}")

        # Timestamps
        if status.get('started_at'):
            lines.append(f"\n시작: {status['started_at']}")

        return "\n".join(lines)

    # ========== 미구현 핸들러 추가 ==========

    async def _handle_compare_models(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle compare_models action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        models_to_compare = config.get("models_to_compare", [])

        if not models_to_compare or len(models_to_compare) < 2:
            return {
                "new_state": ConversationState.SELECTING_MODEL,
                "message": "비교할 모델을 2개 이상 지정해주세요.\n\n예시: resnet50과 efficientnet_b0 비교해줘",
                "temp_data": temp_data
            }

        try:
            from app.utils.tool_registry import tool_registry

            # Call compare_models tool
            result = await tool_registry.call_tool(
                "compare_models",
                {"model_specs": models_to_compare},
                self.db,
                user_id=None
            )

            # Format comparison results
            message = f"**모델 비교 결과:**\n\n"

            for i, model in enumerate(models_to_compare, 1):
                message += f"{i}. {model.get('framework')} / {model.get('name')}\n"

            message += f"\n{result.get('comparison', '모델 비교 기능은 곧 제공됩니다.')}"

            temp_data["model_comparison"] = result

            return {
                "new_state": ConversationState.COMPARING_MODELS,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to compare models: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"모델 비교 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_resume_training(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle resume_training action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        job_id = config.get("job_id")

        if not job_id:
            return {
                "new_state": ConversationState.GATHERING_CONFIG,
                "message": "재개할 학습 작업의 ID를 알려주세요.\n\n예시: job 123 재개해줘",
                "temp_data": temp_data
            }

        try:
            # Check if job exists and is stopped
            job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

            if not job:
                return {
                    "new_state": ConversationState.ERROR,
                    "message": f"학습 작업 {job_id}를 찾을 수 없습니다.",
                    "temp_data": temp_data
                }

            if job.status != "stopped":
                return {
                    "new_state": ConversationState.MONITORING_TRAINING,
                    "message": f"학습 작업 {job_id}는 현재 '{job.status}' 상태입니다. 중지된 작업만 재개할 수 있습니다.",
                    "temp_data": temp_data
                }

            # TODO: Implement actual resume logic
            # For now, just return a message indicating it's not implemented yet
            message = f"**학습 재개:**\n\n"
            message += f"Job ID {job_id}의 학습을 재개합니다.\n"
            message += f"모델: {job.model_name}\n"
            message += f"프레임워크: {job.framework}\n\n"
            message += "학습 재개 기능은 곧 구현됩니다."

            return {
                "new_state": ConversationState.MONITORING_TRAINING,
                "message": message,
                "temp_data": temp_data
            }

        except Exception as e:
            logger.error(f"Failed to resume training: {str(e)}", exc_info=True)
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 재개 중 오류가 발생했습니다: {str(e)}",
                "temp_data": temp_data
            }

    async def _handle_start_batch_inference(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle start_batch_inference action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        job_id = config.get("job_id")
        image_dir = config.get("image_dir")

        if not job_id:
            return {
                "new_state": ConversationState.GATHERING_CONFIG,
                "message": "추론에 사용할 학습 작업 ID를 알려주세요.\n\n예시: job 123으로 추론해줘",
                "temp_data": temp_data
            }

        if not image_dir:
            return {
                "new_state": ConversationState.GATHERING_CONFIG,
                "message": "추론할 이미지 폴더 경로를 알려주세요.\n\n예시: C:/datasets/images 폴더 추론해줘",
                "temp_data": temp_data
            }

        # Check if job exists
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not job:
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 작업 {job_id}를 찾을 수 없습니다.",
                "temp_data": temp_data
            }

        # TODO: Implement actual batch inference logic
        message = f"**배치 추론 시작:**\n\n"
        message += f"Job ID: {job_id}\n"
        message += f"이미지 폴더: {image_dir}\n"
        message += f"모델: {job.model_name}\n\n"
        message += "배치 추론 기능은 곧 구현됩니다."

        return {
            "new_state": ConversationState.RUNNING_INFERENCE,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_show_inference_results(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_inference_results action"""
        temp_data = session.temp_data or {}

        # Check if inference results exist in temp_data
        inference_results = temp_data.get("inference_results")

        if not inference_results:
            return {
                "new_state": ConversationState.IDLE,
                "message": "표시할 추론 결과가 없습니다. 먼저 추론을 실행해주세요.",
                "temp_data": temp_data
            }

        # Format inference results
        message = f"**추론 결과:**\n\n"

        job_id = inference_results.get("job_id")
        if job_id:
            message += f"Job ID: {job_id}\n\n"

        predictions = inference_results.get("predictions", [])
        if predictions:
            message += f"총 {len(predictions)}개 이미지 추론 완료:\n\n"
            for i, pred in enumerate(predictions[:10], 1):  # Show first 10
                image_name = pred.get("image", f"image_{i}")
                pred_class = pred.get("class", "Unknown")
                confidence = pred.get("confidence", 0)
                message += f"{i}. {image_name}: {pred_class} ({confidence:.2%})\n"

            if len(predictions) > 10:
                message += f"\n... 외 {len(predictions) - 10}개"
        else:
            message += "추론 결과가 비어있습니다."

        return {
            "new_state": ConversationState.VIEWING_RESULTS,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_show_validation_results(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_validation_results action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        job_id = config.get("job_id")

        if not job_id:
            return {
                "new_state": ConversationState.GATHERING_CONFIG,
                "message": "검증 결과를 조회할 학습 작업 ID를 알려주세요.\n\n예시: job 123 검증 결과 보여줘",
                "temp_data": temp_data
            }

        # Check if job exists
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not job:
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 작업 {job_id}를 찾을 수 없습니다.",
                "temp_data": temp_data
            }

        # TODO: Fetch validation results from database or MLflow
        message = f"**검증 결과:**\n\n"
        message += f"Job ID: {job_id}\n"
        message += f"모델: {job.model_name}\n"
        message += f"작업 유형: {job.task_type}\n\n"

        if job.final_accuracy:
            message += f"최종 정확도: {job.final_accuracy:.2%}\n"

        message += "\n상세한 검증 메트릭 조회 기능은 곧 구현됩니다."

        return {
            "new_state": ConversationState.VIEWING_RESULTS,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_show_confusion_matrix(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_confusion_matrix action"""
        temp_data = session.temp_data or {}
        config = temp_data.get("config", {})

        job_id = config.get("job_id")

        if not job_id:
            return {
                "new_state": ConversationState.GATHERING_CONFIG,
                "message": "Confusion Matrix를 조회할 학습 작업 ID를 알려주세요.\n\n예시: job 123 confusion matrix 보여줘",
                "temp_data": temp_data
            }

        # Check if job exists
        job = self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not job:
            return {
                "new_state": ConversationState.ERROR,
                "message": f"학습 작업 {job_id}를 찾을 수 없습니다.",
                "temp_data": temp_data
            }

        if job.task_type not in ["classification", "image_classification"]:
            return {
                "new_state": ConversationState.VIEWING_RESULTS,
                "message": f"Confusion Matrix는 분류 작업({job.task_type})에만 사용할 수 있습니다.",
                "temp_data": temp_data
            }

        # TODO: Generate and fetch confusion matrix from validation results
        message = f"**Confusion Matrix:**\n\n"
        message += f"Job ID: {job_id}\n"
        message += f"모델: {job.model_name}\n\n"
        message += "Confusion Matrix 시각화 기능은 곧 구현됩니다."

        return {
            "new_state": ConversationState.VIEWING_RESULTS,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_show_help(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle show_help action"""
        temp_data = session.temp_data or {}

        message = """**Vision AI Training Platform 도움말**

📊 **데이터셋 관리:**
- 데이터셋 분석: "C:/datasets/coco8 분석해줘"
- 데이터셋 목록: "데이터셋 목록 보여줘"

🤖 **모델 선택:**
- 모델 검색: "classification 모델 찾아줘"
- 모델 추천: "어떤 모델이 좋을까?"
- 모델 정보: "yolov8n 정보 알려줘"
- 모델 비교: "resnet50과 efficientnet_b0 비교해줘"

🚀 **학습 관리:**
- 학습 시작: "resnet50으로 학습 시작"
- 학습 상태: "job 123 상태 알려줘"
- 학습 중지: "job 123 중지해줘"
- 학습 재개: "job 123 재개해줘"
- 작업 목록: "학습 작업 목록 보여줘"

🔮 **추론:**
- 빠른 추론: "job 123으로 C:/test/cat.jpg 추론해줘"
- 배치 추론: "job 123으로 C:/test/images 폴더 추론해줘"
- 결과 조회: "추론 결과 보여줘"

📈 **결과 조회:**
- 검증 결과: "job 123 검증 결과 보여줘"
- Confusion Matrix: "job 123 confusion matrix 보여줘"

⚙️ **기타:**
- 도움말: "도움말" 또는 "help"
- 대화 초기화: "초기화" 또는 "reset"

💡 **팁:** 자연어로 편하게 말씀하세요! 예: "고양이 강아지 분류 모델 만들고 싶어"
"""

        return {
            "new_state": ConversationState.IDLE,
            "message": message,
            "temp_data": temp_data
        }

    async def _handle_reset_conversation(
        self,
        action_response: GeminiActionResponse,
        session: SessionModel,
        user_message: str
    ) -> Dict[str, Any]:
        """Handle reset_conversation action"""
        # Clear all temp_data
        session.temp_data = {}

        message = "대화를 초기화했습니다. 새로운 작업을 시작해주세요.\n\n"
        message += "무엇을 도와드릴까요?\n"
        message += "- 데이터셋 분석\n"
        message += "- 모델 검색/추천\n"
        message += "- 학습 시작\n"
        message += "- 추론 실행\n\n"
        message += "도움이 필요하시면 '도움말'을 입력해주세요."

        return {
            "new_state": ConversationState.INITIAL,
            "message": message,
            "temp_data": {}
        }
