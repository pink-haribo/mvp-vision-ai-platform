# 문제 분석 및 해결 방안

## 발견한 사실

### 1. Config 저장 패턴
- **test_simple.py (단일 메시지)**: ✅ 작동
  - 메시지: `"C:\\datasets\\cls\\imagenet-10"`
  - 결과: dataset_path 정상 저장

- **test_debug.py (멀티스텝)**: ❌ 실패
  - Step 1: "ResNet18로 학습하고 싶어" → framework, model_name, task_type 저장됨
  - Step 2: "C:\\datasets\\cls\\imagenet-10" → dataset_path 저장 안됨
  - Step 3: "기본값으로 해줘" → epochs, batch_size, learning_rate 저장 안됨

### 2. 로깅 문제
- Fallback 로그 파일이 전혀 생성되지 않음
- logger.warning도 출력되지 않음
- **결론**: `handle_action`이 호출되지 않거나, 코드 변경이 반영되지 않음

### 3. DB 확인 결과
- API는 200 OK 응답
- DB State는 변경됨 (gathering_config → selecting_project)
- 하지만 temp_data.config는 Step 1 이후 변경 없음

## 가능한 원인

### 가설 1: LLM이 current_config를 반환하지 않음
```python
# LLM이 이렇게 반환해야 하는데:
{
  "action": "ask_clarification",
  "current_config": {
    "framework": "timm",
    "model_name": "resnet18",
    "task_type": "image_classification",
    "dataset_path": "C:\\datasets\\cls\\imagenet-10"  # 새로 추가
  }
}

# 실제로는 이렇게 반환:
{
  "action": "ask_clarification",
  "current_config": null  # 또는 빈 객체
}
```

### 가설 2: Fallback 로직이 실행되지 않음
- 코드 변경이 uvicorn reload에 의해 반영되지 않음
- Python bytecode 캐시 문제
- Import 경로 문제

### 가설 3: test_simple.py는 다른 경로로 작동
- LLM이 단일 메시지에서 dataset_path를 직접 파싱
- Fallback 없이도 작동

## 즉시 적용 가능한 해결책

### Option A: LLM 프롬프트 강화 (권장)
`llm_structured.py`의 프롬프트를 수정:
```python
**CRITICAL RULE**:
When returning current_config, you MUST include ALL previously collected fields.
NEVER drop any field that was set in previous steps.

Example:
Previous config: {"framework": "timm", "model_name": "resnet18"}
User says: "C:\\datasets\\cls\\imagenet-10"
You MUST return:
{
  "current_config": {
    "framework": "timm",          # KEEP
    "model_name": "resnet18",      # KEEP
    "dataset_path": "C:\\datasets\\cls\\imagenet-10"  # ADD
  }
}
```

### Option B: 백엔드 강제 병합 (이미 시도했지만 작동 안함)
- handle_action에서 강제 merge
- 문제: 코드 변경이 반영되지 않는 것 같음

### Option C: 임시 Workaround - 한 번에 입력
사용자에게 모든 정보를 한 번에 입력하도록 안내:
```
"ResNet18로 C:\\datasets\\cls\\imagenet-10 데이터셋을 사용해서 50 에포크, 배치 32, 학습률 0.001로 학습해줘"
```

### Option D: 프론트엔드 폼 추가 (가장 확실)
```tsx
<ConfigForm
  onSubmit={(config) => {
    // 직접 API로 전송, LLM 우회
    createTrainingJob(config)
  }}
/>
```

## 다음 단계 제안

### 즉시 (지금):
1. ✅ LLM 프롬프트 강화 (Option A)
2. ✅ 전체 flow를 한 번에 입력하는 테스트

### 단기 (오늘):
3. 프론트엔드에 수동 입력 폼 추가
4. 학습 실행 로직 구현 시작

### 중기 (내일~):
5. LLM 추가 개선 (GPT-4 테스트)
6. Agent framework 고려

## 결론

**문제의 본질**: LLM이 current_config에 이전 값을 포함시키지 않음

**즉시 해결 방법**:
1. LLM 프롬프트 개선
2. 또는 프론트엔드 폼 추가 (확실한 방법)

**권장 사항**:
지금은 프론트엔드 폼을 추가해서 학습 진입을 확보하고,
LLM 개선은 학습 기능 구현과 병행
