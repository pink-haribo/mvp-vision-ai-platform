# Vision Platform 디자인 시스템

## 목차
- [개요](#개요)
- [디자인 원칙](#디자인-원칙)
- [컬러 시스템](#컬러-시스템)
- [타이포그래피](#타이포그래피)
- [간격 시스템](#간격-시스템)
- [컴포넌트](#컴포넌트)
- [애니메이션](#애니메이션)
- [다크모드](#다크모드)

## 개요

Vision AI Platform의 디자인 시스템은 **일관성**, **접근성**, **확장성**을 핵심 가치로 합니다.

### 디자인 철학

- **Modern & Clean**: 깔끔하고 현대적인 미니멀 디자인
- **AI-focused**: AI/Vision 느낌의 보라-핑크 그라디언트
- **Professional**: 전문가를 위한 도구의 신뢰감
- **Data-driven**: 데이터와 메트릭의 가독성 최우선

### 영감

- **Vercel**: 미니멀하고 개발자 친화적
- **Linear**: 효율적이고 빠른 인터랙션
- **Stripe**: 명확하고 전문적인 데이터 표현

---

## 디자인 원칙

### 1. 명확성 (Clarity)

모든 UI 요소는 그 목적이 명확해야 합니다.

```
✅ 좋은 예: "학습 시작" 버튼 - 명확한 액션
❌ 나쁜 예: "Go" 버튼 - 모호한 액션
```

### 2. 일관성 (Consistency)

동일한 패턴을 반복적으로 사용합니다.

```
✅ 모든 카드: 12px border-radius, 1px border
✅ 모든 버튼: 8px border-radius, 200ms transition
```

### 3. 피드백 (Feedback)

사용자 액션에 즉각적인 피드백을 제공합니다.

```
✅ 버튼 클릭 → 시각적 변화 (scale, 색상)
✅ 로딩 상태 → spinner 표시
✅ 성공/실패 → 명확한 메시지
```

### 4. 접근성 (Accessibility)

WCAG 2.1 AA 이상을 준수합니다.

```
✅ 색상 대비: 최소 4.5:1
✅ 키보드 네비게이션 지원
✅ 스크린 리더 호환
```

---

## 컬러 시스템

### Brand Colors

#### Primary (Violet)

AI와 Intelligence를 상징하는 보라색

```css
violet-50:  #f5f3ff
violet-100: #ede9fe
violet-200: #ddd6fe
violet-300: #c4b5fd
violet-400: #a78bfa
violet-500: #8b5cf6  /* Main brand color */
violet-600: #7c3aed
violet-700: #6d28d9
violet-800: #5b21b6
violet-900: #4c1d95
```

**사용 예:**
- Primary 버튼 배경: `violet-600`
- Hover: `violet-700`
- Active: `violet-800`
- Badge, Tag: `violet-100` + `violet-700` (text)

#### Secondary (Fuchsia)

Creativity와 Vision을 상징하는 핑크

```css
fuchsia-50:  #fdf4ff
fuchsia-100: #fae8ff
fuchsia-200: #f5d0fe
fuchsia-300: #f0abfc
fuchsia-400: #e879f9
fuchsia-500: #d946ef
fuchsia-600: #c026d3  /* Accent color */
fuchsia-700: #a21caf
fuchsia-800: #86198f
fuchsia-900: #701a75
```

**사용 예:**
- Gradient: `violet-600` → `fuchsia-600`
- Accent 요소
- Hover 효과

### Neutral Colors

UI의 기본 베이스

```css
gray-50:  #fafafa  /* Light background */
gray-100: #f5f5f5  /* Card background */
gray-200: #e5e5e5  /* Border */
gray-300: #d4d4d4  /* Disabled state */
gray-400: #a3a3a3
gray-500: #737373
gray-600: #525252  /* Secondary text */
gray-700: #404040
gray-800: #262626  /* Sidebar, dark areas */
gray-900: #171717  /* Primary text */
gray-950: #0a0a0a  /* Darkest */
```

### Semantic Colors

#### Success (Emerald)

```css
emerald-100: #d1fae5  /* Light background */
emerald-500: #10b981  /* Main */
emerald-700: #047857  /* Dark */
```

**사용:**
- 학습 완료
- 검증 성공
- 저장 완료

#### Warning (Amber)

```css
amber-100: #fef3c7
amber-500: #f59e0b
amber-700: #d97706
```

**사용:**
- 주의 필요
- 리소스 부족 경고
- 데이터 불균형

#### Error (Red)

```css
red-100: #fee2e2
red-500: #ef4444
red-700: #dc2626
```

**사용:**
- 학습 실패
- 에러 메시지
- 삭제 액션

#### Info (Blue)

```css
blue-100: #dbeafe
blue-500: #3b82f6
blue-700: #1d4ed8
```

**사용:**
- 정보 메시지
- Tooltip
- Help text

### Gradients

#### Neural Gradient

```css
background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%);
```

AI/Neural Network 느낌의 시안-블루

#### Vision Gradient

```css
background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
```

Vision/Creativity 느낌의 바이올렛-핑크

#### Brand Gradient (Primary)

```css
background: linear-gradient(135deg, #7c3aed 0%, #c026d3 100%);
```

메인 브랜드 그라디언트

---

## 타이포그래피

### Font Family

```css
font-family: 'SUIT', -apple-system, BlinkMacSystemFont, 
             'Segoe UI', sans-serif;
```

**SUIT 폰트 특징:**
- 한글/영문 조화
- Variable Font (100-900)
- 높은 가독성
- 현대적 느낌

### Font Scale

```css
text-xs:   0.75rem   /* 12px - Caption, Label */
text-sm:   0.875rem  /* 14px - Body small, Secondary */
text-base: 1rem      /* 16px - Body, Default */
text-lg:   1.125rem  /* 18px - Subheading */
text-xl:   1.25rem   /* 20px - Heading 4 */
text-2xl:  1.5rem    /* 24px - Heading 3 */
text-3xl:  1.875rem  /* 30px - Heading 2 */
text-4xl:  2.25rem   /* 36px - Heading 1 */
text-5xl:  3rem      /* 48px - Display */
```

### Font Weight

```css
font-normal:    400  /* Body text */
font-medium:    500  /* Emphasis */
font-semibold:  600  /* Subheadings */
font-bold:      700  /* Headings */
font-extrabold: 800  /* Display, Hero */
```

### Line Height

```css
leading-tight:   1.25  /* Headings */
leading-normal:  1.5   /* Body */
leading-relaxed: 1.75  /* Long-form content */
```

### Usage Examples

```tsx
// Heading 1
<h1 className="text-4xl font-bold text-gray-900">
  Vision AI Platform
</h1>

// Heading 2
<h2 className="text-3xl font-bold text-gray-900">
  학습 진행상황
</h2>

// Heading 3
<h3 className="text-2xl font-semibold text-gray-900">
  메트릭
</h3>

// Body
<p className="text-base text-gray-700 leading-normal">
  자연어로 모델을 설명하면 자동으로 학습합니다.
</p>

// Small text
<span className="text-sm text-gray-600">
  2시간 전
</span>

// Caption
<span className="text-xs text-gray-500 uppercase tracking-wider">
  최근 프로젝트
</span>
```

---

## 간격 시스템

### Spacing Scale (4px 단위)

```css
0:  0       /* 0px */
1:  0.25rem /* 4px */
2:  0.5rem  /* 8px */
3:  0.75rem /* 12px */
4:  1rem    /* 16px */
5:  1.25rem /* 20px */
6:  1.5rem  /* 24px */
8:  2rem    /* 32px */
10: 2.5rem  /* 40px */
12: 3rem    /* 48px */
16: 4rem    /* 64px */
20: 5rem    /* 80px */
24: 6rem    /* 96px */
```

### 사용 가이드

#### 컴포넌트 내부 간격

```css
/* 작은 간격 */
padding: 0.5rem;  /* 8px - Badge, Small button */

/* 기본 간격 */
padding: 1rem;    /* 16px - Button, Input */

/* 중간 간격 */
padding: 1.5rem;  /* 24px - Card, Modal */

/* 큰 간격 */
padding: 2rem;    /* 32px - Section */
```

#### 컴포넌트 간 간격

```css
/* 밀접 관계 */
gap: 0.5rem;   /* 8px - Button group, Tags */

/* 관련 요소 */
gap: 1rem;     /* 16px - Form fields, List items */

/* 섹션 구분 */
gap: 2rem;     /* 32px - Sections */

/* 페이지 레벨 */
gap: 4rem;     /* 64px - Major sections */
```

---

## 컴포넌트

### Border Radius

```css
rounded-none: 0
rounded-sm:   0.375rem  /* 6px - Input */
rounded-md:   0.5rem    /* 8px - Button */
rounded-lg:   0.75rem   /* 12px - Card */
rounded-xl:   1rem      /* 16px - Large card */
rounded-2xl:  1.5rem    /* 24px - Modal */
rounded-full: 9999px    /* Avatar, Badge */
```

### Shadows

```css
/* Subtle elevation */
shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05)

/* Default card */
shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1)

/* Elevated element */
shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1)

/* Modal, Popover */
shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1)

/* Special effects */
shadow-glow-violet: 0 0 20px rgba(139, 92, 246, 0.4)
shadow-glow-emerald: 0 0 20px rgba(16, 185, 129, 0.4)
```

### Buttons

#### Primary Button

```tsx
<button className="
  px-4 py-2.5 
  bg-violet-600 hover:bg-violet-700 active:bg-violet-800
  text-white font-semibold
  rounded-lg
  shadow-md hover:shadow-lg
  transition-all duration-200
  disabled:opacity-40 disabled:cursor-not-allowed
">
  학습 시작
</button>
```

#### Secondary Button

```tsx
<button className="
  px-4 py-2.5
  bg-gray-200 hover:bg-gray-300 active:bg-gray-400
  text-gray-900 font-semibold
  rounded-lg
  shadow-sm
  transition-all duration-200
">
  취소
</button>
```

#### Ghost Button

```tsx
<button className="
  px-4 py-2.5
  bg-transparent hover:bg-gray-100 active:bg-gray-200
  text-gray-700 font-semibold
  rounded-lg
  border border-gray-300
  transition-all duration-200
">
  설정
</button>
```

#### Gradient Button

```tsx
<button className="
  px-6 py-3
  bg-gradient-to-r from-violet-600 to-fuchsia-600
  hover:from-violet-700 hover:to-fuchsia-700
  text-white font-bold
  rounded-lg
  shadow-lg hover:shadow-xl
  transition-all duration-200
">
  새 프로젝트
</button>
```

### Cards

```tsx
<div className="
  bg-white
  border-2 border-gray-200
  rounded-xl
  p-6
  shadow-md
  hover:shadow-lg
  transition-all duration-200
">
  <h3 className="text-lg font-bold text-gray-900 mb-2">
    Card Title
  </h3>
  <p className="text-sm text-gray-600">
    Card content
  </p>
</div>
```

### Input Fields

```tsx
<input className="
  w-full
  px-4 py-2.5
  border-2 border-gray-300
  rounded-lg
  text-sm font-medium
  focus:outline-none
  focus:ring-4 focus:ring-violet-500/30
  focus:border-violet-600
  transition-all duration-200
" />
```

### Badges

```tsx
/* Primary */
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-violet-200 text-violet-900
  text-xs font-semibold
  rounded-full
">
  학습중
</span>

/* Success */
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-emerald-200 text-emerald-900
  text-xs font-semibold
  rounded-full
">
  완료
</span>

/* With dot */
<span className="
  inline-flex items-center gap-1.5
  px-2.5 py-1
  bg-violet-200 text-violet-900
  text-xs font-semibold
  rounded-full
">
  <span className="w-1.5 h-1.5 bg-violet-900 rounded-full animate-pulse" />
  학습중
</span>
```

---

## 애니메이션

### Duration

```css
duration-100: 100ms  /* Instant feedback */
duration-200: 200ms  /* Default transitions */
duration-300: 300ms  /* Smooth transitions */
duration-500: 500ms  /* Slow, noticeable */
```

### Easing

```css
ease-linear:  linear
ease-in:      cubic-bezier(0.4, 0, 1, 1)
ease-out:     cubic-bezier(0, 0, 0.2, 1)
ease-in-out:  cubic-bezier(0.4, 0, 0.2, 1)  /* Default */
```

### Keyframe Animations

#### Fade In

```css
@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}

.animate-fade-in {
  animation: fadeIn 200ms ease-out;
}
```

#### Slide Up

```css
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-slide-up {
  animation: slideUp 300ms ease-out;
}
```

#### Scale In

```css
@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.animate-scale-in {
  animation: scaleIn 200ms ease-out;
}
```

#### Pulse

```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.animate-pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
```

#### Spinner

```css
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.animate-spin {
  animation: spin 1s linear infinite;
}
```

### Usage Examples

```tsx
/* Button hover */
<button className="
  transform hover:scale-105
  transition-transform duration-200
">
  Hover me
</button>

/* Card entrance */
<div className="animate-slide-up">
  Card content
</div>

/* Loading dot */
<span className="
  w-2 h-2 bg-violet-600 rounded-full
  animate-pulse
" />

/* Spinner */
<svg className="animate-spin h-5 w-5">
  <circle ... />
</svg>
```

---

## 다크모드

### 색상 매핑

```css
/* Light mode (default) */
bg-white → bg-gray-900
bg-gray-50 → bg-gray-800
text-gray-900 → text-gray-100
border-gray-200 → border-gray-700

/* Dark mode */
.dark {
  /* Background */
  --bg-primary: #0a0a0a;
  --bg-secondary: #171717;
  --bg-tertiary: #262626;
  
  /* Text */
  --text-primary: #fafafa;
  --text-secondary: #a3a3a3;
  --text-tertiary: #737373;
  
  /* Border */
  --border-color: #404040;
}
```

### 구현

```tsx
/* Tailwind Dark Mode */
<div className="
  bg-white dark:bg-gray-900
  text-gray-900 dark:text-gray-100
  border-gray-200 dark:border-gray-700
">
  Content
</div>

/* 버튼 */
<button className="
  bg-violet-600 dark:bg-violet-500
  hover:bg-violet-700 dark:hover:bg-violet-600
  text-white
">
  Button
</button>
```

---

## 접근성 체크리스트

### 색상 대비

- [ ] 모든 텍스트: 최소 4.5:1 대비
- [ ] 대형 텍스트 (18px+): 최소 3:1 대비
- [ ] UI 컴포넌트: 최소 3:1 대비

### 키보드 네비게이션

- [ ] Tab으로 모든 요소 접근 가능
- [ ] Focus 상태 명확히 표시
- [ ] Enter/Space로 액션 실행 가능

### 스크린 리더

- [ ] 모든 이미지에 alt 텍스트
- [ ] 버튼에 aria-label
- [ ] 폼 필드에 label 연결

### 기타

- [ ] 폰트 크기 최소 14px
- [ ] 터치 타겟 최소 44x44px
- [ ] 애니메이션 prefers-reduced-motion 지원

---

## 다음 단계

- [컴포넌트 라이브러리](UI_COMPONENTS.md)
- [디자인 토큰 코드](../lib/design-tokens.ts)
- [Tailwind 설정](../tailwind.config.js)
- [Figma 파일](https://figma.com/...)
