# 권한 시스템 가이드

## 개요

Vision AI Platform은 **2단계 권한 시스템**을 사용합니다:
1. **시스템 레벨 권한** - 전체 플랫폼에 대한 권한
2. **프로젝트 레벨 권한** - 개별 프로젝트에 대한 권한

---

## 1. 시스템 레벨 권한

### User 모델의 권한 필드

```python
class User:
    is_active: bool        # 활성화 여부
    is_superuser: bool     # 슈퍼 관리자 여부
```

### 권한 종류

#### 1.1 일반 사용자 (Regular User)
- `is_active: True`
- `is_superuser: False`

**권한:**
- ✅ 자신의 프로젝트 생성/수정/삭제
- ✅ 자신의 학습 작업 생성/실행
- ✅ 자신의 데이터 조회
- ✅ 초대받은 프로젝트 접근 (역할에 따라)
- ❌ 다른 사용자 데이터 접근 불가
- ❌ 시스템 설정 변경 불가

#### 1.2 슈퍼 관리자 (Superuser)
- `is_active: True`
- `is_superuser: True`

**권한:**
- ✅ **모든 일반 사용자 권한**
- ✅ 모든 프로젝트 조회/수정/삭제
- ✅ 모든 사용자 관리 (생성/수정/삭제)
- ✅ 시스템 설정 변경
- ✅ 시스템 로그 조회
- ✅ 사용자 계정 활성화/비활성화

#### 1.3 비활성 사용자 (Inactive User)
- `is_active: False`

**권한:**
- ❌ 로그인 불가
- ❌ 모든 API 접근 불가

---

## 2. 프로젝트 레벨 권한

### ProjectMember 모델

```python
class ProjectMember:
    project_id: int        # 프로젝트 ID
    user_id: int           # 사용자 ID
    role: str              # 역할: owner, admin, member, viewer
    invited_by: int        # 초대한 사용자 ID
    joined_at: datetime    # 참여 일시
```

### 역할 종류

#### 2.1 Owner (소유자)
- 프로젝트를 생성한 사용자 (자동 부여)
- **모든 권한 보유**

**권한:**
- ✅ 프로젝트 삭제
- ✅ 프로젝트 정보 수정
- ✅ 멤버 초대/제거
- ✅ 멤버 역할 변경
- ✅ 학습 작업 생성/실행/중단/삭제
- ✅ 실험 결과 조회/다운로드
- ✅ 데이터셋 업로드/삭제

#### 2.2 Admin (관리자)
- Owner가 지정한 관리자

**권한:**
- ✅ 프로젝트 정보 수정
- ✅ 멤버 초대/제거 (Owner 제외)
- ✅ 학습 작업 생성/실행/중단/삭제
- ✅ 실험 결과 조회/다운로드
- ✅ 데이터셋 업로드/삭제
- ❌ 프로젝트 삭제 불가
- ❌ Owner 역할 변경 불가

#### 2.3 Member (멤버)
- 프로젝트에 초대받은 일반 멤버

**권한:**
- ✅ 학습 작업 생성/실행
- ✅ 자신이 만든 학습 작업 중단/삭제
- ✅ 실험 결과 조회/다운로드
- ✅ 데이터셋 업로드
- ❌ 프로젝트 정보 수정 불가
- ❌ 멤버 관리 불가
- ❌ 다른 멤버의 학습 작업 중단/삭제 불가

#### 2.4 Viewer (조회자)
- 읽기 전용 멤버

**권한:**
- ✅ 프로젝트 정보 조회
- ✅ 실험 결과 조회
- ✅ 학습 로그 조회
- ❌ 학습 작업 생성/실행 불가
- ❌ 데이터 수정/삭제 불가
- ❌ 데이터셋 업로드 불가

---

## 3. 권한 체크 구현

### 3.1 FastAPI Dependencies

```python
from app.utils.dependencies import (
    get_current_user,           # 현재 로그인 사용자
    get_current_active_user,    # 활성화된 사용자
    get_current_superuser       # 슈퍼 관리자
)

# 사용 예시
@router.get("/admin/users")
async def list_all_users(
    current_user: User = Depends(get_current_superuser)  # 슈퍼유저만 접근 가능
):
    ...

@router.get("/projects")
async def list_my_projects(
    current_user: User = Depends(get_current_active_user)  # 활성 사용자만 접근 가능
):
    ...
```

### 3.2 프로젝트 권한 체크

```python
def check_project_permission(project_id: int, user_id: int, db: Session) -> bool:
    """프로젝트 접근 권한 확인"""
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return False

    # Owner 체크
    if project.user_id == user_id:
        return True

    # Member 체크
    member = db.query(ProjectMember).filter(
        ProjectMember.project_id == project_id,
        ProjectMember.user_id == user_id
    ).first()

    return member is not None

# 사용 예시
@router.get("/projects/{project_id}")
async def get_project(
    project_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    if not check_project_permission(project_id, current_user.id, db):
        raise HTTPException(403, "You don't have permission to access this project")
    ...
```

### 3.3 역할별 권한 체크

```python
from enum import Enum

class ProjectRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

def get_user_project_role(project_id: int, user_id: int, db: Session) -> str | None:
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

    return member.role if member else None

def require_project_role(required_role: ProjectRole):
    """특정 역할 이상을 요구하는 의존성"""
    role_hierarchy = {
        ProjectRole.VIEWER: 0,
        ProjectRole.MEMBER: 1,
        ProjectRole.ADMIN: 2,
        ProjectRole.OWNER: 3,
    }

    async def check_role(
        project_id: int,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
    ):
        user_role = get_user_project_role(project_id, current_user.id, db)

        if not user_role:
            raise HTTPException(403, "You don't have access to this project")

        if role_hierarchy[user_role] < role_hierarchy[required_role]:
            raise HTTPException(403, f"Requires {required_role} role or higher")

        return current_user

    return check_role

# 사용 예시
@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: int,
    current_user: User = Depends(require_project_role(ProjectRole.OWNER))
):
    # OWNER만 삭제 가능
    ...

@router.post("/projects/{project_id}/training")
async def create_training(
    project_id: int,
    current_user: User = Depends(require_project_role(ProjectRole.MEMBER))
):
    # MEMBER 이상(MEMBER, ADMIN, OWNER)만 학습 작업 생성 가능
    ...
```

---

## 4. 권한 매트릭스

| 작업 | Viewer | Member | Admin | Owner | Superuser |
|-----|--------|--------|-------|-------|-----------|
| **시스템** |
| 다른 사용자 데이터 조회 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 사용자 관리 | ❌ | ❌ | ❌ | ❌ | ✅ |
| 시스템 설정 변경 | ❌ | ❌ | ❌ | ❌ | ✅ |
| **프로젝트** |
| 프로젝트 조회 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 프로젝트 수정 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 프로젝트 삭제 | ❌ | ❌ | ❌ | ✅ | ✅ |
| 멤버 초대 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 멤버 제거 | ❌ | ❌ | ✅* | ✅ | ✅ |
| 역할 변경 | ❌ | ❌ | ✅* | ✅ | ✅ |
| **학습 작업** |
| 학습 작업 생성 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 학습 작업 실행 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 자신의 작업 중단 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 타인의 작업 중단 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 학습 작업 삭제 | ❌ | ✅** | ✅ | ✅ | ✅ |
| **데이터** |
| 데이터셋 조회 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 데이터셋 업로드 | ❌ | ✅ | ✅ | ✅ | ✅ |
| 데이터셋 삭제 | ❌ | ❌ | ✅ | ✅ | ✅ |
| 실험 결과 조회 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 실험 결과 다운로드 | ✅ | ✅ | ✅ | ✅ | ✅ |

*Admin은 Owner를 제외한 멤버만 관리 가능
**Member는 자신이 만든 작업만 삭제 가능

---

## 5. 보안 모범 사례

### 5.1 JWT 토큰
- Access Token: 1시간 만료
- Refresh Token: 7일 만료
- 토큰은 localStorage에 저장 (XSS 방지 필요)

### 5.2 비밀번호
- 최소 8자 이상
- bcrypt 해싱 (rounds=12)
- 평문 비밀번호는 절대 저장하지 않음

### 5.3 API 보안
- 모든 엔드포인트에 인증 적용
- 슈퍼유저 외 다른 사용자 데이터 접근 불가
- 프로젝트별 권한 체크 필수

### 5.4 감사 로그 (추후 구현)
- 중요 작업(삭제, 권한 변경 등) 로그 기록
- 로그인/로그아웃 기록
- 비정상 접근 시도 기록

---

## 6. 기본 Admin 계정

Migration 실행 시 자동 생성되는 기본 관리자:

```
Email: admin@example.com
Password: admin123
Role: Superuser
```

⚠️ **중요**: 프로덕션 환경에서는 반드시 비밀번호를 변경하세요!

---

## 7. 향후 확장 계획

- [ ] API Key 인증 지원
- [ ] OAuth2 (Google, GitHub) 로그인
- [ ] 2FA (Two-Factor Authentication)
- [ ] IP 화이트리스트
- [ ] 세션 관리 (강제 로그아웃)
- [ ] 권한 템플릿 (역할 프리셋)
- [ ] 감사 로그 시스템
- [ ] Rate Limiting
