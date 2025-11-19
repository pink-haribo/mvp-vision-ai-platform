# 새로운 권한 시스템

## 📋 변경 요약

### 1. 조직 구조
**3단계 계층**: 회사 → 사업부 → 부서

| 필드 | 타입 | 옵션 |
|------|------|------|
| company | 선택 | 삼성전자, 협력사, 직접입력 |
| company_custom | 텍스트 | company="직접입력" 선택 시 입력 |
| division | 선택 | 생산기술연구소, MX, VD, DA, SR, 직접입력 |
| division_custom | 텍스트 | division="직접입력" 선택 시 입력 |
| department | 텍스트 | 자유 입력 |

### 2. 시스템 레벨 권한 (5단계)

```python
class SystemRole:
    GUEST = "guest"                          # 기본 모델만 사용
    STANDARD_ENGINEER = "standard_engineer"  # 모든 모델 사용 가능
    ADVANCED_ENGINEER = "advanced_engineer"  # 세부 기능 사용 가능
    MANAGER = "manager"                      # 권한 승급 가능
    ADMIN = "admin"                          # 모든 기능
```

#### 권한 상세

| Role | 기본 모델 | 고급 모델 | 세부 기능 | 권한 승급 | 사용자 관리 | 프로젝트 관리 |
|------|-----------|-----------|-----------|-----------|-------------|---------------|
| Guest | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Standard Engineer | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Advanced Engineer | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Manager | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Admin | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**기능 매핑:**
- **기본 모델**: ResNet-18, ResNet-50
- **고급 모델**: EfficientNet, YOLO, Custom Models
- **세부 기능**: 하이퍼파라미터 튜닝, 분산 학습, 모델 export
- **권한 승급**: Guest → Standard → Advanced 승급 가능
- **사용자 관리**: 사용자 조회, 비활성화, 권한 변경
- **프로젝트 관리**: 모든 프로젝트 조회/수정/삭제

### 3. 프로젝트 레벨 권한 (2단계 - 단순화)

```python
class ProjectRole:
    MEMBER = "member"  # 프로젝트 멤버
    OWNER = "owner"    # 프로젝트 소유자
```

#### 권한 상세

| 작업 | Member | Owner |
|------|--------|-------|
| **프로젝트** |
| 프로젝트 정보 조회 | ✅ | ✅ |
| 프로젝트 정보 수정 | ❌ | ✅ |
| 프로젝트 삭제 | ❌ | ✅ |
| **멤버 관리** |
| 멤버 초대 | ❌ | ✅ |
| 멤버 제거 | ❌ | ✅ |
| 멤버를 Owner로 승급 | ❌ | ✅ |
| **학습 작업** |
| 학습 작업 생성 | ✅ | ✅ |
| 학습 작업 실행/중단 | ✅ | ✅ |
| 학습 작업 삭제 | ✅* | ✅ |
| **데이터** |
| 데이터셋 조회 | ✅ | ✅ |
| 데이터셋 업로드 | ✅ | ✅ |
| 데이터셋 삭제 | ❌ | ✅ |
| 실험 결과 조회 | ✅ | ✅ |
| 실험 결과 다운로드 | ✅ | ✅ |

*Member는 자신이 만든 작업만 삭제 가능

---

## 🔄 Migration 변경사항

### User 테이블 필드

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),

    -- 조직 정보
    company VARCHAR(100),
    company_custom VARCHAR(255),
    division VARCHAR(100),
    division_custom VARCHAR(255),
    department VARCHAR(255),

    -- 연락처 & 소개
    phone_number VARCHAR(50),
    bio TEXT,

    -- 권한 & 상태
    system_role VARCHAR(50) NOT NULL DEFAULT 'guest',
    is_active BOOLEAN NOT NULL DEFAULT 1,

    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
)
```

**제거된 필드:**
- `is_superuser` → `system_role = 'admin'`으로 대체

---

## 📝 회원가입 폼 구조

```typescript
{
  // 기본 정보
  email: "user@example.com",
  password: "password123",
  full_name: "홍길동",

  // 조직 정보
  company: "삼성전자" | "협력사" | "직접입력",
  company_custom: "ABC 주식회사",  // company="직접입력" 시에만
  division: "MX" | "VD" | ... | "직접입력",
  division_custom: "디스플레이",   // division="직접입력" 시에만
  department: "AI 개발팀",

  // 연락처
  phone_number: "010-1234-5678",
  bio: "컴퓨터 비전 엔지니어"
}
```

---

## 🔐 권한 체크 함수

### 시스템 레벨 권한

```python
from app.schemas.enums import SystemRole, SYSTEM_ROLE_HIERARCHY

def require_system_role(required_role: SystemRole):
    """시스템 권한 체크"""
    async def check_role(current_user: User = Depends(get_current_user)):
        if SYSTEM_ROLE_HIERARCHY[current_user.system_role] < SYSTEM_ROLE_HIERARCHY[required_role]:
            raise HTTPException(403, f"Requires {required_role} or higher")
        return current_user
    return check_role

# 사용 예시
@router.post("/advanced-training")
async def create_advanced_training(
    current_user: User = Depends(require_system_role(SystemRole.ADVANCED_ENGINEER))
):
    # Advanced Engineer 이상만 접근 가능
    ...

@router.post("/users/{user_id}/promote")
async def promote_user(
    user_id: int,
    current_user: User = Depends(require_system_role(SystemRole.MANAGER))
):
    # Manager 이상만 권한 승급 가능
    ...
```

### 프로젝트 레벨 권한

```python
from app.schemas.enums import ProjectRole, PROJECT_ROLE_HIERARCHY

def get_user_project_role(project_id: int, user_id: int, db: Session) -> ProjectRole | None:
    """사용자의 프로젝트 역할 조회"""
    project = db.query(Project).filter(Project.id == project_id).first()

    # Owner 체크
    if project and project.user_id == user_id:
        return ProjectRole.OWNER

    # Member 체크
    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user_id
    ).first()

    return ProjectRole(member.role) if member else None

def require_project_role(required_role: ProjectRole):
    """프로젝트 권한 체크"""
    async def check_role(
        project_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        user_role = get_user_project_role(project_id, current_user.id, db)

        if not user_role:
            raise HTTPException(403, "You don't have access to this project")

        if PROJECT_ROLE_HIERARCHY[user_role] < PROJECT_ROLE_HIERARCHY[required_role]:
            raise HTTPException(403, f"Requires project {required_role} role")

        return current_user
    return check_role

# 사용 예시
@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(require_project_role(ProjectRole.OWNER))
):
    # OWNER만 프로젝트 삭제 가능
    ...

@router.post("/projects/{project_id}/training")
async def create_training(
    project_id: int,
    current_user: User = Depends(require_project_role(ProjectRole.MEMBER))
):
    # MEMBER 이상(MEMBER, OWNER) 학습 작업 생성 가능
    ...
```

---

## 🎯 기본 사용자 설정

| Email | Password | System Role | 용도 |
|-------|----------|-------------|------|
| admin@example.com | admin123 | admin | 시스템 관리자 |

**⚠️ 프로덕션에서 반드시 비밀번호 변경 필요!**

---

## 🔮 향후 확장

- [ ] 권한 승급 워크플로우 (요청 → 승인)
- [ ] 프로젝트 템플릿 (권한 프리셋)
- [ ] 감사 로그 (권한 변경 기록)
- [ ] 배치 권한 관리 (CSV import)
- [ ] 팀 단위 권한 관리

---

## ✅ TODO

- [x] SystemRole enum 정의
- [x] ProjectRole 단순화 (member/owner)
- [x] User 스키마 업데이트
- [x] Migration 스크립트 업데이트
- [ ] auth.py register 함수 수정
- [ ] dependencies.py 권한 체크 함수 추가
- [ ] 회원가입 폼 업데이트 (dropdown)
- [ ] 권한 가이드 재작성
