# Database Migrations

수동 데이터베이스 마이그레이션 스크립트.

## Usage

```bash
cd platform/backend
venv/Scripts/python.exe migrations/migrate_xxx.py
```

## Migration Order

마이그레이션은 순서대로 실행해야 합니다. 각 스크립트는 이미 실행된 마이그레이션을 건너뜁니다.

## Note

이 폴더의 스크립트들은 Alembic과 별개로 동작하는 수동 마이그레이션입니다.
Alembic 마이그레이션은 `alembic/` 폴더에 있습니다.
