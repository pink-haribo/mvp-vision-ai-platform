# Appendices (부록)

추가 참고 자료, 용어 정의, 설계 결정 기록 등을 제공합니다.

## 목차

### [7.1 Key Files Reference](./key-files-reference.md)
주요 파일 위치 및 역할 종합 표

**내용**:
- Backend 주요 파일
- Frontend 주요 파일
- Training 주요 파일
- Docker 관련 파일
- Database 파일

**활용**: 빠른 파일 검색 및 코드 참조

---

### [7.2 Glossary](./glossary.md)
플랫폼 용어 정의

**내용**:
- Adapter
- Execution Mode
- Callbacks
- Primary Metric
- MLflow Run/Experiment
- 기타 기술 용어

**활용**: 문서 이해 및 팀 커뮤니케이션

---

### [7.3 Architecture Decision Records (ADR)](./adr.md)
주요 설계 결정 및 그 이유

**내용**:
- ADR-001: Why Adapter Pattern?
- ADR-002: Why Docker Image Separation?
- ADR-003: Why SQLite + MLflow?
- ADR-004: Why Subprocess + Docker Dual Mode?

**활용**: 설계 의도 이해, 리팩토링 시 참조

---

### [7.4 Related Documents](./related-docs.md)
기존 문서 링크 및 설명

**내용**:
- Architecture 문서
- Planning 문서
- Feature 설계 문서
- 외부 참고 자료

**활용**: 상세 설계 내용 참조

---

## 언제 참조하나요?

- **용어가 헷갈릴 때** → Glossary
- **파일 위치를 모를 때** → Key Files Reference
- **설계 의도가 궁금할 때** → ADR
- **더 자세한 내용이 필요할 때** → Related Documents

---

[← 돌아가기](../README.md)
