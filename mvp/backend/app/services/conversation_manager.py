"""
Conversation Manager (Phase 1+2)

Orchestrates conversation flow:
1. Parse user message with LLM (get structured action)
2. Execute action via ActionHandlers
3. Update session state and temp_data
4. Save messages to DB
5. Return response
"""

import logging
import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.models import Session as SessionModel, Message as MessageModel
from app.models.conversation import ConversationState, GeminiActionResponse, ActionType
from app.utils.llm_structured import structured_intent_parser
from app.services.action_handlers import ActionHandlers

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation flow with state machine + structured actions

    This replaces the old text-parsing approach with:
    - Explicit state management
    - LLM structured output (Gemini)
    - Action-based execution
    """

    def __init__(self, db: Session):
        self.db = db
        self.action_handlers = ActionHandlers(db)

    async def process_message(
        self,
        session_id: int,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process user message through conversation flow

        Args:
            session_id: Current session ID
            user_message: User's message text

        Returns:
            dict: {
                "message": str,              # Response to user
                "state": str,                # New conversation state
                "training_job_id": int       # Optional, if training started
            }
        """
        try:
            # 1. Load session
            session = self.db.query(SessionModel).filter(
                SessionModel.id == session_id
            ).first()

            if not session:
                raise ValueError(f"Session {session_id} not found")

            current_state = ConversationState(session.state)
            temp_data = session.temp_data or {}

            # TRACE: Step 1 - Load from DB
            import sys
            sys.stderr.write(f"\n[TRACE-1-LOAD] Session {session_id}\n")
            sys.stderr.write(f"  State: {current_state}\n")
            sys.stderr.write(f"  temp_data from DB: {str(temp_data)}\n")
            if "config" in temp_data:
                sys.stderr.write(f"  config keys: {list(temp_data['config'].keys())}\n")
            sys.stderr.flush()

            logger.info(f"[Session {session_id}] Current state: {current_state}")
            logger.debug(f"[Session {session_id}] Temp data: {json.dumps(temp_data, ensure_ascii=False)}")

            # 2. Build conversation context (last 5 messages)
            context = self._build_context(session)

            # 3. Handle simple option selection in SELECTING_PROJECT state
            # (LLM sometimes fails to parse simple "1", "2", "3" inputs correctly)
            if current_state == ConversationState.SELECTING_PROJECT:
                direct_action = self._handle_project_selection(user_message, temp_data)
                if direct_action:
                    logger.info(f"[Session {session_id}] Direct action (bypassing LLM): {direct_action.action}")
                    action_response = direct_action
                else:
                    # Fall back to LLM for complex inputs
                    action_response = await structured_intent_parser.parse_intent(
                        user_message=user_message,
                        state=current_state,
                        context=context,
                        temp_data=temp_data
                    )
            else:
                # 3. Parse intent with LLM (get structured action)
                logger.info(f"[Session {session_id}] Parsing intent: '{user_message[:100]}...'")

                action_response: GeminiActionResponse = await structured_intent_parser.parse_intent(
                    user_message=user_message,
                    state=current_state,
                    context=context,
                    temp_data=temp_data
                )

            logger.info(f"[Session {session_id}] LLM action: {action_response.action}")
            logger.debug(f"[Session {session_id}] LLM message: {action_response.message}")

            # EXPLICIT DEBUG - Log LLM response to file
            debug_log_path = "C:/Users/flyto/Project/Github/mvp-vision-ai-platform/mvp/backend/llm_debug.log"
            with open(debug_log_path, "a", encoding="utf-8") as f:
                f.write("\n" + "="*80 + "\n")
                f.write(f"[DEBUG] LLM Response for session {session_id}:\n")
                f.write(f"Action: {action_response.action}\n")
                f.write(f"Message: {action_response.message[:200]}\n")
                if action_response.current_config:
                    f.write(f"Current Config: {json.dumps(action_response.current_config, ensure_ascii=False)}\n")
                else:
                    f.write(f"Current Config: NONE/NULL\n")
                if action_response.config:
                    f.write(f"Config: {json.dumps(action_response.config, ensure_ascii=False)}\n")
                f.write("="*80 + "\n")

            # 4. Execute action
            logger.warning(f"[CM] Calling handle_action for action: {action_response.action}")
            logger.warning(f"[CM] User message: {user_message}")
            result = await self.action_handlers.handle_action(
                action_response=action_response,
                session=session,
                user_message=user_message
            )
            logger.warning(f"[CM] handle_action returned")

            new_state = result["new_state"]
            response_message = result["message"]
            updated_temp_data = result["temp_data"]
            training_job_id = result.get("training_job_id")

            logger.info(f"[Session {session_id}] State transition: {current_state} -> {new_state}")

            # TRACE: Step 5 - Before saving to DB
            print(f"\n[TRACE-5-SAVE] Saving to DB:")
            print(f"  new_state: {new_state}")
            print(f"  updated_temp_data: {json.dumps(updated_temp_data, ensure_ascii=False)}")
            if "config" in updated_temp_data:
                print(f"  config keys: {list(updated_temp_data['config'].keys())}")

            # 5. Update session in DB
            session.state = new_state.value
            session.temp_data = updated_temp_data

            # CRITICAL FIX: Force SQLAlchemy to detect JSON column change
            # When updated_temp_data is the same dict object, SQLAlchemy won't see the change
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(session, "temp_data")

            # 6. Save messages
            # Save user message
            user_msg = MessageModel(
                session_id=session.id,
                role="user",
                content=user_message
            )
            self.db.add(user_msg)

            # Save assistant message
            assistant_msg = MessageModel(
                session_id=session.id,
                role="assistant",
                content=response_message
            )
            self.db.add(assistant_msg)

            # 7. Commit all changes
            self.db.commit()

            # TRACE: Step 6 - After commit, verify what's in DB
            self.db.refresh(session)
            print(f"\n[TRACE-6-VERIFY] After commit:")
            print(f"  session.state: {session.state}")
            print(f"  session.temp_data: {session.temp_data}")
            if session.temp_data and "config" in session.temp_data:
                print(f"  config keys in DB: {list(session.temp_data['config'].keys())}")
            else:
                print(f"  NO CONFIG IN DB!")

            logger.info(f"[Session {session_id}] Response sent, state updated to {new_state}")

            # 8. Return response
            response = {
                "message": response_message,
                "state": new_state.value,
            }

            if training_job_id:
                response["training_job_id"] = training_job_id
                logger.info(f"[Session {session_id}] Training job created: {training_job_id}")

            # Include selected_project_id for frontend to show project detail
            selected_project_id = result.get("selected_project_id")
            if selected_project_id:
                response["selected_project_id"] = selected_project_id
                logger.info(f"[Session {session_id}] Project selected: {selected_project_id}")

            # Phase 1: Include action-specific data for frontend
            if "dataset_analysis" in updated_temp_data:
                response["dataset_analysis"] = updated_temp_data["dataset_analysis"]
            if "model_search_results" in updated_temp_data:
                response["model_search_results"] = updated_temp_data["model_search_results"]
            if "recommended_models" in updated_temp_data:
                response["recommended_models"] = updated_temp_data["recommended_models"]
            if "available_datasets" in updated_temp_data:
                response["available_datasets"] = updated_temp_data["available_datasets"]
            if "training_status" in updated_temp_data:
                response["training_status"] = updated_temp_data["training_status"]
            if "inference_results" in updated_temp_data:
                response["inference_results"] = updated_temp_data["inference_results"]

            return response

        except Exception as e:
            logger.error(f"[Session {session_id}] Error processing message: {e}", exc_info=True)

            # Save error message
            try:
                error_msg = MessageModel(
                    session_id=session_id,
                    role="assistant",
                    content=f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                )
                self.db.add(error_msg)
                self.db.commit()
            except Exception as db_error:
                logger.error(f"Failed to save error message: {db_error}")

            return {
                "message": f"죄송합니다. 오류가 발생했습니다: {str(e)}",
                "state": ConversationState.ERROR.value
            }

    def _handle_project_selection(self, user_message: str, temp_data: Dict[str, Any]) -> Optional[GeminiActionResponse]:
        """
        Handle simple project selection inputs (1, 2, 3) directly

        Returns GeminiActionResponse if input matches a simple pattern, None otherwise
        """
        msg = user_message.strip().lower()

        # Check if we're already showing project list (user is selecting from list)
        if "available_projects" in temp_data:
            # User is selecting a specific project from the list by number
            # Handle numeric selection directly
            if msg.replace("번", "").isdigit():
                project_number = msg.replace("번", "")
                return GeminiActionResponse(
                    action=ActionType.SELECT_PROJECT,
                    message=f"프로젝트 {project_number}번을 선택합니다...",
                    project_identifier=project_number
                )
            # Otherwise, let LLM handle name-based selection
            return None

        # We're at the initial selection screen (신규/기존/건너뛰기)

        # Option 1: Create new project
        if msg in ["1", "1번", "신규"]:
            return GeminiActionResponse(
                action=ActionType.ASK_CLARIFICATION,
                message="신규 프로젝트를 생성합니다. 프로젝트 이름을 입력해주세요.\n\n예시: 이미지 분류 프로젝트 - 설명",
                missing_fields=["project_name"]
            )

        # Option 2: Select existing project
        if msg in ["2", "2번", "기존"]:
            return GeminiActionResponse(
                action=ActionType.SHOW_PROJECT_LIST,
                message="기존 프로젝트를 조회합니다..."
            )

        # Option 3: Skip project
        if msg in ["3", "3번", "건너뛰기", "없이"]:
            return GeminiActionResponse(
                action=ActionType.SKIP_PROJECT,
                message="프로젝트 없이 진행합니다."
            )

        # Not a simple option - let LLM handle it
        return None

    def _build_context(self, session: SessionModel, max_messages: int = 10) -> str:
        """
        Build conversation context from recent messages

        Args:
            session: Current session
            max_messages: Maximum number of messages to include

        Returns:
            str: Formatted conversation history
        """
        messages = (
            self.db.query(MessageModel)
            .filter(MessageModel.session_id == session.id)
            .order_by(MessageModel.created_at.desc())
            .limit(max_messages)
            .all()
        )

        # Reverse to chronological order
        messages = list(reversed(messages))

        if not messages:
            return ""

        context_lines = []
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_lines.append(f"{role}: {msg.content}")

        return "\n".join(context_lines)

    async def create_new_session(self) -> SessionModel:
        """
        Create a new conversation session

        Returns:
            SessionModel: Newly created session
        """
        new_session = SessionModel(
            state=ConversationState.INITIAL.value,
            temp_data={}
        )
        self.db.add(new_session)
        self.db.commit()
        self.db.refresh(new_session)

        logger.info(f"Created new session: {new_session.id}")
        return new_session

    def get_session_info(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Get session information

        Args:
            session_id: Session ID

        Returns:
            dict: Session info or None if not found
        """
        session = self.db.query(SessionModel).filter(
            SessionModel.id == session_id
        ).first()

        if not session:
            return None

        # Count messages
        message_count = self.db.query(MessageModel).filter(
            MessageModel.session_id == session.id
        ).count()

        return {
            "id": session.id,
            "state": session.state,
            "temp_data": session.temp_data,
            "message_count": message_count,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
        }

    def reset_session(self, session_id: int) -> bool:
        """
        Reset session to initial state (clear conversation)

        Args:
            session_id: Session ID to reset

        Returns:
            bool: True if successful, False if session not found
        """
        session = self.db.query(SessionModel).filter(
            SessionModel.id == session_id
        ).first()

        if not session:
            return False

        # Reset state and temp_data
        session.state = ConversationState.INITIAL.value
        session.temp_data = {}

        # Optional: Delete messages (or keep for history)
        # self.db.query(MessageModel).filter(
        #     MessageModel.session_id == session.id
        # ).delete()

        self.db.commit()

        logger.info(f"Reset session {session_id} to initial state")
        return True
