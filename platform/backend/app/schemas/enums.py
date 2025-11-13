"""Enum types for the application."""

from enum import Enum


class SystemRole(str, Enum):
    """
    System-level user roles.

    These roles control what features users can access across the entire platform.
    """
    GUEST = "guest"                           # 기본 모델만 사용
    STANDARD_ENGINEER = "standard_engineer"   # 모든 모델 사용 가능
    ADVANCED_ENGINEER = "advanced_engineer"   # 세부 기능 사용 가능
    MANAGER = "manager"                       # 권한 승급 가능
    ADMIN = "admin"                           # 모든 기능 (권한/사용자/프로젝트 관리)


class ProjectRole(str, Enum):
    """
    Project-level user roles.

    Simple two-tier system for project collaboration.
    """
    MEMBER = "member"   # 프로젝트 멤버 (학습 작업 생성/실행)
    OWNER = "owner"     # 프로젝트 소유자 (멤버 초대, Owner 승급 가능)


class Company(str, Enum):
    """Predefined company options."""
    SAMSUNG = "삼성전자"
    PARTNER = "협력사"
    CUSTOM = "직접 입력"


class Division(str, Enum):
    """Predefined division options."""
    PRODUCTION_TECH = "생산기술연구소"
    MX = "MX"
    VD = "VD"
    DA = "DA"
    SR = "SR"
    CUSTOM = "직접 입력"


# Role hierarchy for permission checking
SYSTEM_ROLE_HIERARCHY = {
    SystemRole.GUEST: 0,
    SystemRole.STANDARD_ENGINEER: 1,
    SystemRole.ADVANCED_ENGINEER: 2,
    SystemRole.MANAGER: 3,
    SystemRole.ADMIN: 4,
}

PROJECT_ROLE_HIERARCHY = {
    ProjectRole.MEMBER: 0,
    ProjectRole.OWNER: 1,
}
