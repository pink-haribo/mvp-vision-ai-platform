# MVP 디자인 구현 가이드

AI 코딩에서 통일성 있고 세련된 UI를 만들기 위한 MVP 전용 디자인 가이드입니다.

## 🎯 목적

AI가 코드를 생성할 때 다음을 보장:
1. **일관된 스타일** - 모든 화면이 동일한 디자인 언어 사용
2. **명확한 레퍼런스** - 복사 가능한 코드 스니펫
3. **제약 조건** - MVP 범위 내에서만 사용

---

## 📐 MVP 화면 레이아웃

### 메인 페이지 (단일 페이지)

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (h-16)                                                    │
│  Vision AI Platform (MVP) │ Status Badge                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Main Content (flex)                                              │
│  ┌───────────────────────────┐  ┌──────────────────────────────┐│
│  │ Left: Chat Panel          │  │ Right: Training Panel         ││
│  │ (w-1/2, max-w-2xl)        │  │ (w-1/2, max-w-2xl)            ││
│  │                           │  │                                ││
│  │ ┌─────────────────────┐   │  │ ┌──────────────────────────┐ ││
│  │ │ Message List        │   │  │ │ Status Card              │ ││
│  │ │ (flex-1, overflow)  │   │  │ │                          │ ││
│  │ │                     │   │  │ │  Status: Running         │ ││
│  │ │ User: "ResNet50..." │   │  │ │  ██████████░░░░  65%     │ ││
│  │ │                     │   │  │ │                          │ ││
│  │ │ AI: "좋아요..."     │   │  │ │  Epoch: 65/100           │ ││
│  │ │                     │   │  │ │  Loss: 0.234             │ ││
│  │ │                     │   │  │ │  Accuracy: 89.2%         │ ││
│  │ └─────────────────────┘   │  │ └──────────────────────────┘ ││
│  │                           │  │                                ││
│  │ ┌─────────────────────┐   │  │ ┌──────────────────────────┐ ││
│  │ │ Input Box           │   │  │ │ Actions                  │ ││
│  │ │ Type message...     │   │  │ │ [Stop] [Download Model]  │ ││
│  │ └─────────────────────┘   │  │ └──────────────────────────┘ ││
│  └───────────────────────────┘  └──────────────────────────────┘│
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 MVP 전용 컬러 팔레트

### 사용 가능한 색상 (제한)

```typescript
// Primary - 브랜드
'violet-600'  // 주요 액션 버튼
'violet-700'  // Hover
'violet-100'  // 배경 강조

// Neutral - 기본 UI
'gray-50'     // 페이지 배경
'gray-100'    // 카드 배경
'gray-200'    // Border
'gray-600'    // Secondary text
'gray-900'    // Primary text

// Semantic
'emerald-600' // Success (완료, 성공)
'amber-600'   // Warning (주의)
'red-600'     // Error (실패, 삭제)

// Gradient (특별한 경우만)
'bg-gradient-to-r from-violet-600 to-fuchsia-600' // 강조 버튼
```

**⚠️ 주의:** 이 외의 색상은 사용하지 않습니다!

---

## 🔤 타이포그래피 규칙

### 사용 가능한 크기

```typescript
// Headings
'text-2xl font-bold'      // H1 (페이지 제목)
'text-xl font-semibold'   // H2 (섹션 제목)
'text-lg font-semibold'   // H3 (카드 제목)

// Body
'text-base'               // 기본 텍스트
'text-sm text-gray-600'   // Secondary 텍스트
'text-xs text-gray-500'   // Caption, 라벨
```

### 예시

```tsx
<h1 className="text-2xl font-bold text-gray-900">
  Vision AI Platform
</h1>

<h2 className="text-xl font-semibold text-gray-900">
  학습 진행상황
</h2>

<p className="text-base text-gray-700">
  자연어로 모델을 설명하면 자동으로 학습합니다.
</p>

<span className="text-sm text-gray-600">
  2분 전
</span>
```

---

## 🧩 MVP 컴포넌트 스타일

### Button (3가지만)

```tsx
// Primary - 주요 액션
<button className="
  px-4 py-2.5
  bg-violet-600 hover:bg-violet-700
  text-white font-semibold
  rounded-lg shadow-md
  transition-all duration-200
  disabled:opacity-40
">
  학습 시작
</button>

// Secondary - 보조 액션
<button className="
  px-4 py-2.5
  bg-gray-200 hover:bg-gray-300
  text-gray-900 font-semibold
  rounded-lg
  transition-all duration-200
">
  취소
</button>

// Danger - 위험한 액션
<button className="
  px-4 py-2.5
  bg-red-600 hover:bg-red-700
  text-white font-semibold
  rounded-lg
  transition-all duration-200
">
  중단
</button>
```

### Card (1가지만)

```tsx
<div className="
  bg-white
  border-2 border-gray-200
  rounded-xl
  p-6
  shadow-md
">
  <h3 className="text-lg font-semibold text-gray-900 mb-4">
    카드 제목
  </h3>
  <div className="space-y-2">
    {/* Content */}
  </div>
</div>
```

### Input (1가지만)

```tsx
<div className="space-y-1.5">
  <label className="text-sm font-semibold text-gray-900">
    라벨
  </label>
  <input
    className="
      w-full
      px-4 py-2.5
      border-2 border-gray-300
      rounded-lg
      text-sm
      focus:outline-none
      focus:ring-4 focus:ring-violet-500/20
      focus:border-violet-600
      transition-all duration-200
    "
    placeholder="입력하세요..."
  />
</div>
```

### Badge (3가지만)

```tsx
// Running
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-violet-100 text-violet-900
  text-xs font-semibold
  rounded-full
">
  <span className="w-1.5 h-1.5 bg-violet-600 rounded-full animate-pulse" />
  실행 중
</span>

// Success
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-emerald-100 text-emerald-900
  text-xs font-semibold
  rounded-full
">
  완료
</span>

// Error
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-red-100 text-red-900
  text-xs font-semibold
  rounded-full
">
  실패
</span>
```

### Progress Bar (1가지만)

```tsx
<div className="space-y-2">
  <div className="flex justify-between text-sm">
    <span className="font-semibold text-gray-900">Training Progress</span>
    <span className="text-gray-600">65%</span>
  </div>
  <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
    <div
      className="h-full bg-violet-600 rounded-full transition-all duration-500"
      style={{ width: '65%' }}
    />
  </div>
</div>
```

---

## 📏 간격 규칙

### 일관된 간격만 사용

```typescript
// 컴포넌트 내부
'p-4'      // 작은 패딩 (16px)
'p-6'      // 기본 패딩 (24px)
'p-8'      // 큰 패딩 (32px)

// 컴포넌트 간
'space-y-2'   // 작은 간격 (8px)
'space-y-4'   // 기본 간격 (16px)
'space-y-6'   // 큰 간격 (24px)

// Gap (Flex/Grid)
'gap-2'    // 8px
'gap-4'    // 16px
'gap-6'    // 24px
```

---

## 🎬 애니메이션 (간단하게만)

```typescript
// Transitions (모든 인터랙티브 요소)
'transition-all duration-200'

// 특별한 애니메이션 (최소한으로)
'animate-pulse'     // 로딩 dot
'animate-spin'      // Spinner
```

---

## 📱 레이아웃 구조

### 메인 페이지 구조

```tsx
// app/page.tsx
<div className="min-h-screen bg-gray-50">
  {/* Header */}
  <header className="h-16 bg-white border-b-2 border-gray-200">
    <div className="h-full max-w-7xl mx-auto px-6 flex items-center justify-between">
      <h1 className="text-xl font-bold text-gray-900">
        Vision AI Platform (MVP)
      </h1>
      <Badge variant="primary">Beta</Badge>
    </div>
  </header>

  {/* Main */}
  <main className="h-[calc(100vh-4rem)] max-w-7xl mx-auto p-6">
    <div className="h-full flex gap-6">
      {/* Left: Chat */}
      <div className="w-1/2 flex flex-col">
        <ChatPanel />
      </div>

      {/* Right: Training */}
      <div className="w-1/2 flex flex-col">
        <TrainingPanel />
      </div>
    </div>
  </main>
</div>
```

### ChatPanel 구조

```tsx
// components/chat/ChatPanel.tsx
<div className="h-full flex flex-col bg-white border-2 border-gray-200 rounded-xl shadow-md">
  {/* Header */}
  <div className="p-6 border-b-2 border-gray-200">
    <h2 className="text-xl font-semibold text-gray-900">Chat</h2>
  </div>

  {/* Messages */}
  <div className="flex-1 overflow-y-auto p-6 space-y-4">
    {messages.map(message => (
      <Message key={message.id} {...message} />
    ))}
  </div>

  {/* Input */}
  <div className="p-6 border-t-2 border-gray-200">
    <div className="flex gap-2">
      <input
        className="flex-1 px-4 py-2.5 border-2 border-gray-300 rounded-lg"
        placeholder="메시지를 입력하세요..."
      />
      <button className="px-4 py-2.5 bg-violet-600 text-white rounded-lg">
        전송
      </button>
    </div>
  </div>
</div>
```

### TrainingPanel 구조

```tsx
// components/training/TrainingPanel.tsx
<div className="h-full flex flex-col gap-6">
  {/* Status Card */}
  <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-md">
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-lg font-semibold text-gray-900">학습 상태</h3>
      <Badge variant={status === 'running' ? 'primary' : 'success'}>
        {status}
      </Badge>
    </div>

    {/* Progress */}
    <ProgressBar value={progress} label={`Epoch ${currentEpoch}/${totalEpochs}`} />
  </div>

  {/* Metrics Card */}
  <div className="bg-white border-2 border-gray-200 rounded-xl p-6 shadow-md">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">메트릭</h3>
    <div className="space-y-3">
      <MetricRow label="Loss" value="0.234" />
      <MetricRow label="Accuracy" value="89.2%" />
    </div>
  </div>

  {/* Actions */}
  <div className="flex gap-2">
    <button className="flex-1 px-4 py-2.5 bg-red-600 text-white rounded-lg">
      중단
    </button>
    <button className="flex-1 px-4 py-2.5 bg-violet-600 text-white rounded-lg">
      다운로드
    </button>
  </div>
</div>
```

---

## 🎯 아이콘 사용

**아이콘 라이브러리:** [Lucide React](https://lucide.dev/)

```bash
pnpm add lucide-react
```

**사용 가능한 아이콘 (제한):**

```tsx
import {
  Send,           // 전송
  StopCircle,     // 중단
  Download,       // 다운로드
  CheckCircle,    // 성공
  XCircle,        // 에러
  AlertCircle,    // 경고
  Loader2,        // 로딩 (animate-spin 과 함께)
} from 'lucide-react'

// 사용 예시
<Send className="w-4 h-4" />
<Loader2 className="w-4 h-4 animate-spin" />
```

---

## ✅ AI 코딩 시 체크리스트

### 코드 생성 시 반드시 확인:

- [ ] **색상**: violet-600, gray-*, emerald-600, amber-600, red-600만 사용
- [ ] **버튼**: Primary, Secondary, Danger 3가지만
- [ ] **간격**: p-4/p-6/p-8, space-y-2/4/6, gap-2/4/6만 사용
- [ ] **텍스트**: text-xs/sm/base/lg/xl/2xl만 사용
- [ ] **Border Radius**: rounded-lg (버튼/input), rounded-xl (카드)
- [ ] **Transition**: transition-all duration-200
- [ ] **Shadow**: shadow-md (카드)
- [ ] **아이콘**: Lucide React에서만 가져오기

### 금지 사항:

- ❌ 새로운 색상 추가
- ❌ 복잡한 애니메이션
- ❌ 다중 레이아웃
- ❌ 커스텀 그라디언트 (기본 제공 외)
- ❌ 과도한 shadow나 효과

---

## 📦 필수 파일

### 1. Tailwind Config (`tailwind.config.ts`)
→ 이미 생성됨: `mvp/frontend/tailwind.config.ts`

### 2. Utils (`lib/utils/cn.ts`)
→ 이미 생성됨: `mvp/frontend/lib/utils/cn.ts`

### 3. Package.json dependencies

```json
{
  "dependencies": {
    "next": "14.0.4",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "tailwindcss": "3.3.6",
    "clsx": "2.0.0",
    "tailwind-merge": "2.2.0",
    "lucide-react": "0.294.0"
  }
}
```

---

## 🚀 사용 방법

### AI에게 디자인 요청 시:

**✅ 좋은 예:**
```
"MVP_DESIGN_GUIDE.md를 참고해서 ChatPanel 컴포넌트를 만들어줘"
```

**❌ 나쁜 예:**
```
"채팅 화면 만들어줘" (너무 모호함)
```

### 코드 리뷰 시:

MVP_DESIGN_GUIDE.md의 체크리스트 확인!

---

## 📚 참고 문서

- [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md) - 전체 디자인 시스템
- [UI_COMPONENTS.md](UI_COMPONENTS.md) - 상세 컴포넌트 스펙
- [MVP_STRUCTURE.md](MVP_STRUCTURE.md) - MVP 폴더 구조
- [Tailwind CSS Docs](https://tailwindcss.com/docs)
- [Lucide Icons](https://lucide.dev/)

---

**이 가이드를 따르면 AI가 생성하는 모든 코드가 통일된 스타일을 갖습니다! 🎨**
