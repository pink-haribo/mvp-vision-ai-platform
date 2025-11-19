# TODO Documentation

구현 진행 상황 추적을 위한 문서 폴더.

## Structure

```
docs/todo/
├── README.md                      # 이 파일
├── IMPLEMENTATION_TO_DO_LIST.md   # 메인 TODO 리스트 (간결)
└── references/                    # 상세 구현 노트 (추후)
```

## Usage

### IMPLEMENTATION_TO_DO_LIST.md

- **목적**: 전체 구현 진행 상황을 한눈에 파악
- **형식**: 간결한 체크리스트
- **상세 내용**: 각 항목은 reference 문서로 연결

### Reference Documents

기존 상세 문서들은 원래 위치에 유지:
- `docs/planning/` - 기능 계획 문서
- `docs/` - 설계/가이드 문서

`docs/todo/references/`는 기존 폴더에 맞지 않는 상세 구현 노트용.

## Status Icons

- ✅ 완료
- 🔄 진행중
- ⬜ 미시작

## Related Documents

- [MVP_TO_PLATFORM_CHECKLIST.md](../planning/MVP_TO_PLATFORM_CHECKLIST.md) - 상세 진행 로그 (원본)
- [E2E_TEST_GUIDE.md](../E2E_TEST_GUIDE.md) - 테스트 가이드
