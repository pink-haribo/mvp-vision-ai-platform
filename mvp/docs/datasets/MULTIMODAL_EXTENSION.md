# DICE Format: 멀티모달(Vision+Text) 확장

**Version**: v1.1 (Multimodal Extension)
**Date**: 2025-01-04
**Status**: Design Complete

---

## 📋 목차

1. [배경 및 필요성](#배경-및-필요성)
2. [지원할 멀티모달 태스크](#지원할-멀티모달-태스크)
3. [확장된 스키마 설계](#확장된-스키마-설계)
4. [Task별 Annotation 예시](#task별-annotation-예시)
5. [하위 호환성](#하위-호환성)
6. [구현 계획](#구현-계획)

---

## 배경 및 필요성

### 현재 DICE Format v1.0의 한계

v1.0은 순수 비전 태스크만 지원:
- Image Classification
- Object Detection
- Instance Segmentation
- Semantic Segmentation
- Pose Estimation
- Super-Resolution

**문제점**: 텍스트 정보를 포함하는 태스크 미지원
- Image Captioning: 이미지 → 텍스트 설명
- VQA (Visual Question Answering): 이미지 + 질문 → 답변
- Visual Grounding: 이미지 + 텍스트 → Bounding Box
- OCR: 이미지 → 텍스트 + 위치
- Vision-Language Pre-training: 이미지-텍스트 쌍

### v1.1 목표

✅ **텍스트 데이터 통합 저장**
✅ **멀티모달 태스크 8종 지원**
✅ **v1.0 하위 호환성 유지**
✅ **Framework 호환 (HuggingFace datasets, CLIP, BLIP, LLaVA)**

---

## 지원할 멀티모달 태스크

### 1. Image Captioning
**입력**: 이미지
**출력**: 텍스트 설명 (1개 이상)

**예시**:
- 이미지: 고양이가 소파에 앉아있음
- Caption: "A fluffy orange cat sitting on a gray sofa"

**활용 모델**: BLIP-2, GIT, ClipCap

---

### 2. Visual Question Answering (VQA)
**입력**: 이미지 + 질문
**출력**: 답변

**예시**:
- 이미지: 공원에서 뛰어노는 개
- 질문: "What is the dog doing?"
- 답변: "Playing fetch in the park"

**활용 모델**: BLIP-2, LLaVA, InstructBLIP

---

### 3. Visual Grounding (Referring Expression)
**입력**: 이미지 + 텍스트 설명
**출력**: Bounding Box

**예시**:
- 이미지: 여러 사람이 있는 사진
- 설명: "The person wearing a red hat on the left"
- 출력: [100, 50, 200, 300] (bbox)

**활용 모델**: GLIP, MDETR, OWL-ViT

---

### 4. OCR (Optical Character Recognition)
**입력**: 이미지
**출력**: 텍스트 + Bounding Box

**예시**:
- 이미지: 간판 사진
- 출력:
  - "COFFEE SHOP" at [120, 50, 300, 100]
  - "Open 9AM-6PM" at [130, 110, 290, 140]

**활용 모델**: PaddleOCR, TrOCR, Donut

---

### 5. Dense Captioning
**입력**: 이미지
**출력**: Region별 설명 (Bbox + Caption)

**예시**:
- Region 1: [100, 50, 200, 150] → "A red car"
- Region 2: [300, 100, 400, 250] → "A person walking"

**활용 모델**: Dense Captioning models

---

### 6. Image-Text Matching
**입력**: 이미지 + 텍스트
**출력**: Match score (0-1)

**예시**:
- 이미지: 강아지 사진
- 텍스트: "A cute puppy playing with a ball"
- Score: 0.92

**활용 모델**: CLIP, ALIGN

---

### 7. Text-to-Image Retrieval
**입력**: 텍스트 쿼리
**출력**: 관련 이미지 ID 리스트

**예시**:
- 쿼리: "sunset over the ocean"
- 결과: [img_001, img_045, img_123]

**활용 모델**: CLIP-based retrieval

---

### 8. Visual Dialogue
**입력**: 이미지 + 대화 히스토리
**출력**: 응답

**예시**:
- 이미지: 주방 사진
- Q1: "What's on the table?" → A1: "A bowl of fruit"
- Q2: "What kind of fruit?" → A2: "Apples and bananas"

**활용 모델**: VisDial models

---

## 확장된 스키마 설계

### annotations.json 최상위 필드 추가

```json
{
  "format_version": "1.1",  // ← v1.0에서 v1.1로 업그레이드
  "dataset_id": "user123-vqa-dataset",
  "dataset_name": "VQA Dataset v2.0",

  "task_type": "visual_question_answering",  // ← 새로운 태스크 타입

  "modalities": ["image", "text"],  // ← NEW: 사용되는 모달리티

  "text_config": {  // ← NEW: 텍스트 데이터 설정
    "language": "en",  // ko, en, multi
    "tokenizer": "bert-base-uncased",  // Optional
    "max_length": 512,  // Optional
    "vocab_size": 30522  // Optional
  },

  "classes": [/* ... */],
  "images": [/* 확장된 annotation 구조 */],
  "statistics": {/* ... */}
}
```

### 지원 Task Types (확장)

```python
# v1.0 (Pure Vision)
VISION_TASKS = [
    "image_classification",
    "object_detection",
    "instance_segmentation",
    "semantic_segmentation",
    "pose_estimation",
    "super_resolution"
]

# v1.1 (Multimodal: Vision + Text)
MULTIMODAL_TASKS = [
    "image_captioning",
    "visual_question_answering",
    "visual_grounding",
    "ocr",
    "dense_captioning",
    "image_text_matching",
    "text_to_image_retrieval",
    "visual_dialogue"
]
```

---

## Task별 Annotation 예시

### 1. Image Captioning

```json
{
  "id": 1,
  "file_name": "cat_on_sofa.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotation": {
    "captions": [
      {
        "caption_id": 1,
        "text": "A fluffy orange cat sitting on a gray sofa",
        "language": "en",
        "labeled_by": "user123",
        "labeled_at": "2025-01-15T10:00:00Z"
      },
      {
        "caption_id": 2,
        "text": "An orange cat relaxing on a couch",
        "language": "en",
        "labeled_by": "user456",
        "labeled_at": "2025-01-15T10:05:00Z"
      }
    ],
    "primary_caption": "A fluffy orange cat sitting on a gray sofa"  // Optional
  },

  "metadata": {
    "num_captions": 2,
    "avg_caption_length": 42
  }
}
```

---

### 2. Visual Question Answering (VQA)

```json
{
  "id": 2,
  "file_name": "park_scene.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotation": {
    "qa_pairs": [
      {
        "qa_id": 1,
        "question": "What is the dog doing?",
        "answer": "Playing fetch",
        "answer_type": "activity",  // activity, object, color, counting, yes/no
        "confidence": 1.0,
        "labeled_by": "user123"
      },
      {
        "qa_id": 2,
        "question": "How many people are in the image?",
        "answer": "3",
        "answer_type": "counting",
        "confidence": 1.0,
        "labeled_by": "user123"
      },
      {
        "qa_id": 3,
        "question": "Is it daytime?",
        "answer": "yes",
        "answer_type": "yes/no",
        "confidence": 1.0,
        "labeled_by": "user456"
      }
    ]
  },

  "metadata": {
    "num_qa_pairs": 3,
    "answer_types": {
      "activity": 1,
      "counting": 1,
      "yes/no": 1
    }
  }
}
```

---

### 3. Visual Grounding (Referring Expression)

```json
{
  "id": 3,
  "file_name": "people_crowd.jpg",
  "width": 3000,
  "height": 2000,
  "split": "train",

  "annotation": {
    "referring_expressions": [
      {
        "ref_id": 1,
        "expression": "The person wearing a red hat on the left",
        "bbox": [100, 50, 200, 300],
        "bbox_format": "xywh",
        "labeled_by": "user123",
        "labeled_at": "2025-01-15T10:00:00Z"
      },
      {
        "ref_id": 2,
        "expression": "The woman in blue dress holding a phone",
        "bbox": [500, 100, 180, 350],
        "bbox_format": "xywh",
        "labeled_by": "user123",
        "labeled_at": "2025-01-15T10:02:00Z"
      }
    ]
  },

  "metadata": {
    "num_referring_expressions": 2
  }
}
```

---

### 4. OCR (Optical Character Recognition)

```json
{
  "id": 4,
  "file_name": "sign_board.jpg",
  "width": 2400,
  "height": 1600,
  "split": "train",

  "annotation": {
    "text_regions": [
      {
        "text_id": 1,
        "text": "COFFEE SHOP",
        "bbox": [120, 50, 300, 100],
        "bbox_format": "xywh",
        "confidence": 0.98,
        "language": "en",
        "font_size": "large",
        "orientation": 0  // degrees
      },
      {
        "text_id": 2,
        "text": "Open 9AM-6PM",
        "bbox": [130, 110, 290, 140],
        "bbox_format": "xywh",
        "confidence": 0.95,
        "language": "en",
        "font_size": "medium",
        "orientation": 0
      },
      {
        "text_id": 3,
        "text": "매일 영업",
        "bbox": [140, 150, 280, 180],
        "bbox_format": "xywh",
        "confidence": 0.92,
        "language": "ko",
        "font_size": "small",
        "orientation": 0
      }
    ]
  },

  "metadata": {
    "num_text_regions": 3,
    "languages": ["en", "ko"],
    "total_characters": 28
  }
}
```

---

### 5. Dense Captioning

```json
{
  "id": 5,
  "file_name": "street_view.jpg",
  "width": 3000,
  "height": 2000,
  "split": "train",

  "annotation": {
    "region_captions": [
      {
        "region_id": 1,
        "bbox": [100, 50, 200, 150],
        "bbox_format": "xywh",
        "caption": "A red car parked on the street",
        "confidence": 0.95
      },
      {
        "region_id": 2,
        "bbox": [300, 100, 400, 250],
        "bbox_format": "xywh",
        "caption": "A person walking with an umbrella",
        "confidence": 0.92
      },
      {
        "region_id": 3,
        "bbox": [800, 20, 200, 100],
        "bbox_format": "xywh",
        "caption": "A traffic light showing green",
        "confidence": 0.88
      }
    ]
  },

  "metadata": {
    "num_regions": 3,
    "avg_caption_length": 35
  }
}
```

---

### 6. Image-Text Matching

```json
{
  "id": 6,
  "file_name": "puppy_playing.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotation": {
    "positive_captions": [
      {
        "caption_id": 1,
        "text": "A cute puppy playing with a ball in the garden",
        "match_score": 1.0
      },
      {
        "caption_id": 2,
        "text": "A young dog having fun outdoors",
        "match_score": 1.0
      }
    ],
    "negative_captions": [
      {
        "caption_id": 3,
        "text": "A cat sleeping on a bed",
        "match_score": 0.0
      },
      {
        "caption_id": 4,
        "text": "People playing soccer in a park",
        "match_score": 0.0
      }
    ]
  },

  "metadata": {
    "num_positive": 2,
    "num_negative": 2
  }
}
```

---

### 7. Text-to-Image Retrieval

**데이터셋 레벨 구조** (images 배열 외부):

```json
{
  "format_version": "1.1",
  "task_type": "text_to_image_retrieval",

  "queries": [
    {
      "query_id": 1,
      "text": "sunset over the ocean",
      "relevant_image_ids": [5, 12, 34, 67],
      "language": "en"
    },
    {
      "query_id": 2,
      "text": "city skyline at night",
      "relevant_image_ids": [8, 23, 45],
      "language": "en"
    }
  ],

  "images": [
    {
      "id": 5,
      "file_name": "beach_sunset.jpg",
      "annotation": {
        "relevant_queries": [1],  // 역방향 참조
        "tags": ["sunset", "ocean", "beach", "sky"]
      }
    }
  ]
}
```

---

### 8. Visual Dialogue

```json
{
  "id": 8,
  "file_name": "kitchen_scene.jpg",
  "width": 1920,
  "height": 1080,
  "split": "train",

  "annotation": {
    "dialogues": [
      {
        "dialogue_id": 1,
        "turns": [
          {
            "turn_id": 1,
            "question": "What's on the table?",
            "answer": "A bowl of fruit",
            "questioner": "user123"
          },
          {
            "turn_id": 2,
            "question": "What kind of fruit?",
            "answer": "Apples and bananas",
            "questioner": "user123"
          },
          {
            "turn_id": 3,
            "question": "How many apples are there?",
            "answer": "Three red apples",
            "questioner": "user456"
          }
        ],
        "created_at": "2025-01-15T10:00:00Z"
      }
    ]
  },

  "metadata": {
    "num_dialogues": 1,
    "total_turns": 3,
    "avg_turns_per_dialogue": 3.0
  }
}
```

---

## 하위 호환성

### v1.0 → v1.1 Migration

**자동 감지 로직**:

```python
def detect_format_version(annotations: dict) -> str:
    version = annotations.get("format_version", "1.0")

    # v1.0 dataset can be read as-is in v1.1
    if version == "1.0":
        # Pure vision tasks - no migration needed
        return "1.0_compatible"

    # v1.1 with multimodal
    if "modalities" in annotations or "text_config" in annotations:
        return "1.1"

    return "1.0"
```

**하위 호환성 보장**:

✅ v1.0 데이터셋은 v1.1 시스템에서 그대로 동작
✅ v1.1 multimodal 필드는 Optional (없어도 됨)
✅ Pure vision task는 v1.0 스키마 유지
✅ v1.0 → v1.1 업그레이드는 `format_version` 필드만 변경

---

## Framework 호환성

### HuggingFace datasets 변환

```python
from datasets import Dataset, Features, Value, Sequence

# VQA 예시
def convert_to_hf_dataset(dice_annotations: dict) -> Dataset:
    features = Features({
        'image': Image(),
        'question': Value('string'),
        'answer': Value('string'),
        'answer_type': Value('string')
    })

    data = []
    for img in dice_annotations['images']:
        for qa in img['annotation']['qa_pairs']:
            data.append({
                'image': img['file_name'],
                'question': qa['question'],
                'answer': qa['answer'],
                'answer_type': qa['answer_type']
            })

    return Dataset.from_dict(data, features=features)
```

### CLIP / BLIP 형식 변환

```python
# Image-Text Pair 변환
def convert_to_clip_format(dice_annotations: dict):
    """DICE Format → CLIP training pairs"""
    pairs = []

    for img in dice_annotations['images']:
        if 'captions' in img['annotation']:
            for cap in img['annotation']['captions']:
                pairs.append({
                    'image_path': img['file_name'],
                    'caption': cap['text']
                })

    return pairs
```

---

## 구현 계획

### Phase 1: 스키마 확장 (1주)
- [ ] annotations.json 스키마에 multimodal 필드 추가
- [ ] 8가지 멀티모달 태스크 스키마 정의
- [ ] Validation 로직 구현 (pydantic)
- [ ] 예시 데이터셋 생성

### Phase 2: Backend API (1주)
- [ ] Dataset 업로드 시 multimodal 필드 파싱
- [ ] Task type별 validation
- [ ] Text 데이터 저장/조회 API
- [ ] Statistics 계산 (텍스트 길이, QA 페어 수 등)

### Phase 3: Format Converter (2주)
- [ ] DICE → HuggingFace datasets 변환
- [ ] DICE → CLIP/BLIP 형식 변환
- [ ] DICE → VQA v2.0 형식 변환
- [ ] DICE → OCR 형식 변환
- [ ] Cache 메커니즘 (content_hash 기반)

### Phase 4: UI/Labeler (3주)
- [ ] Caption 입력 UI
- [ ] VQA 레이블링 UI (질문+답변)
- [ ] Visual Grounding UI (텍스트 + Bbox)
- [ ] OCR 레이블링 UI
- [ ] Multi-turn dialogue UI

### Phase 5: Training Pipeline (2주)
- [ ] Image Captioning 학습 (BLIP-2)
- [ ] VQA 학습 (LLaVA)
- [ ] Visual Grounding 학습 (GLIP)
- [ ] OCR 학습 (TrOCR)

**총 예상 기간**: 9주 (2개월)

---

## 예시 파일

### 완전한 VQA 데이터셋 예시

파일: `example-v1.1-vqa.json`

```json
{
  "format_version": "1.1",
  "dataset_id": "vqa-demo-001",
  "dataset_name": "VQA Demo Dataset",

  "task_type": "visual_question_answering",
  "modalities": ["image", "text"],

  "text_config": {
    "language": "en",
    "max_question_length": 50,
    "max_answer_length": 20
  },

  "created_at": "2025-01-15T10:00:00Z",
  "version": 1,
  "content_hash": "sha256:vqa123...",

  "classes": [],  // VQA는 클래스 없음

  "splits": {
    "train": 80,
    "val": 15,
    "test": 5
  },

  "images": [
    {
      "id": 1,
      "file_name": "images/park_001.jpg",
      "width": 1920,
      "height": 1080,
      "split": "train",

      "annotation": {
        "qa_pairs": [
          {
            "qa_id": 1,
            "question": "What is the dog doing?",
            "answer": "Playing fetch",
            "answer_type": "activity",
            "confidence": 1.0
          },
          {
            "qa_id": 2,
            "question": "How many people are visible?",
            "answer": "3",
            "answer_type": "counting",
            "confidence": 1.0
          }
        ]
      },

      "metadata": {
        "labeled_by": "user123",
        "labeled_at": "2025-01-15T10:00:00Z",
        "num_qa_pairs": 2
      }
    }
  ],

  "statistics": {
    "total_images": 100,
    "total_qa_pairs": 350,
    "avg_qa_per_image": 3.5,
    "answer_type_distribution": {
      "yes/no": 100,
      "counting": 50,
      "activity": 80,
      "object": 70,
      "color": 30,
      "other": 20
    },
    "avg_question_length": 8.5,
    "avg_answer_length": 2.3
  }
}
```

---

## 요약

### v1.1 주요 변경 사항

| 항목 | v1.0 | v1.1 (Multimodal) |
|------|------|-------------------|
| **지원 태스크** | 순수 비전 6종 | 비전 6종 + 멀티모달 8종 |
| **format_version** | "1.0" | "1.1" |
| **새 필드** | - | `modalities`, `text_config` |
| **Annotation 타입** | 이미지 기반 | 이미지+텍스트 |
| **하위 호환** | N/A | ✅ v1.0 완전 호환 |

### 멀티모달 지원 태스크 8종

1. ✅ Image Captioning
2. ✅ Visual Question Answering (VQA)
3. ✅ Visual Grounding
4. ✅ OCR
5. ✅ Dense Captioning
6. ✅ Image-Text Matching
7. ✅ Text-to-Image Retrieval
8. ✅ Visual Dialogue

---

**Last Updated**: 2025-01-04
**Next Steps**: Phase 1 구현 시작 (스키마 확장 및 Validation)
