# UI 컴포넌트 라이브러리

## 목차
- [개요](#개요)
- [Button](#button)
- [Input](#input)
- [Card](#card)
- [Badge](#badge)
- [ProgressBar](#progressbar)
- [Modal](#modal)
- [Toast](#toast)
- [Dropdown](#dropdown)
- [Layout 컴포넌트](#layout-컴포넌트)

## 개요

모든 컴포넌트는 다음 원칙을 따릅니다:

- **Composable**: 작은 컴포넌트를 조합하여 복잡한 UI 구성
- **Accessible**: WCAG 2.1 AA 준수
- **Typed**: TypeScript로 완전한 타입 안정성
- **Tested**: 모든 컴포넌트는 단위 테스트 포함

---

## Button

### Props

```typescript
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger' | 'gradient';
  size?: 'sm' | 'base' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}
```

### Variants

#### Primary

기본 액션 버튼

```tsx
<Button variant="primary">
  학습 시작
</Button>
```

#### Secondary

보조 액션 버튼

```tsx
<Button variant="secondary">
  취소
</Button>
```

#### Ghost

배경이 없는 버튼

```tsx
<Button variant="ghost">
  더보기
</Button>
```

#### Danger

삭제, 중단 등 위험한 액션

```tsx
<Button variant="danger">
  삭제
</Button>
```

#### Gradient

강조가 필요한 주요 액션

```tsx
<Button variant="gradient">
  새 프로젝트
</Button>
```

### Sizes

```tsx
<Button size="sm">Small</Button>
<Button size="base">Base</Button>
<Button size="lg">Large</Button>
```

### States

#### Loading

```tsx
<Button loading>
  처리 중...
</Button>
```

#### Disabled

```tsx
<Button disabled>
  사용 불가
</Button>
```

#### With Icon

```tsx
<Button icon={<PlayIcon className="w-4 h-4" />}>
  학습 시작
</Button>
```

### Full Width

```tsx
<Button fullWidth>
  전체 너비
</Button>
```

### 구현

```tsx
// components/ui/Button.tsx
import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const buttonVariants = cva(
  "inline-flex items-center justify-center font-semibold transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed",
  {
    variants: {
      variant: {
        primary: "bg-violet-600 hover:bg-violet-700 active:bg-violet-800 text-white shadow-md hover:shadow-lg",
        secondary: "bg-gray-200 hover:bg-gray-300 active:bg-gray-400 text-gray-900 shadow-sm",
        ghost: "bg-transparent hover:bg-gray-100 active:bg-gray-200 text-gray-700 border border-gray-300",
        danger: "bg-red-600 hover:bg-red-700 active:bg-red-800 text-white shadow-md",
        gradient: "bg-gradient-to-r from-violet-600 to-fuchsia-600 hover:from-violet-700 hover:to-fuchsia-700 text-white shadow-lg hover:shadow-xl",
      },
      size: {
        sm: "px-3 py-1.5 text-xs rounded-md gap-1.5",
        base: "px-4 py-2.5 text-sm rounded-lg gap-2",
        lg: "px-6 py-3 text-base rounded-lg gap-2",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "base",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  loading?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, loading, icon, children, fullWidth, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          buttonVariants({ variant, size }),
          fullWidth && "w-full",
          className
        )}
        disabled={loading || props.disabled}
        {...props}
      >
        {loading ? (
          <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        ) : icon}
        {children}
      </button>
    );
  }
);

Button.displayName = "Button";
```

---

## Input

### Props

```typescript
interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  icon?: React.ReactNode;
  size?: 'sm' | 'base' | 'lg';
}
```

### Basic Usage

```tsx
<Input 
  label="이메일"
  type="email"
  placeholder="you@example.com"
/>
```

### With Error

```tsx
<Input 
  label="비밀번호"
  type="password"
  error="비밀번호는 최소 8자 이상이어야 합니다"
/>
```

### With Icon

```tsx
<Input 
  label="검색"
  icon={<SearchIcon className="w-4 h-4" />}
  placeholder="모델 검색..."
/>
```

### With Helper Text

```tsx
<Input 
  label="Epochs"
  type="number"
  helperText="권장: 100-200"
/>
```

### 구현

```tsx
// components/ui/Input.tsx
import React from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
  icon?: React.ReactNode;
  size?: 'sm' | 'base' | 'lg';
}

const sizeClasses = {
  sm: 'px-3 py-1.5 text-sm',
  base: 'px-4 py-2.5 text-sm',
  lg: 'px-5 py-3 text-base',
};

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, helperText, icon, size = 'base', ...props }, ref) => {
    return (
      <div className="flex flex-col gap-1.5">
        {label && (
          <label className="text-sm font-semibold text-gray-900">
            {label}
          </label>
        )}
        
        <div className="relative">
          {icon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400">
              {icon}
            </div>
          )}
          
          <input
            ref={ref}
            className={cn(
              "w-full rounded-lg border-2 font-medium transition-all duration-200",
              icon && "pl-10",
              error 
                ? "border-red-300 focus:border-red-500 focus:ring-4 focus:ring-red-500/20" 
                : "border-gray-300 focus:border-violet-600 focus:ring-4 focus:ring-violet-500/20",
              "focus:outline-none",
              sizeClasses[size],
              className
            )}
            {...props}
          />
        </div>
        
        {error && (
          <span className="text-xs font-semibold text-red-600">{error}</span>
        )}
        
        {helperText && !error && (
          <span className="text-xs text-gray-500">{helperText}</span>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";
```

---

## Card

### Props

```typescript
interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
  padding?: 'none' | 'sm' | 'normal' | 'lg';
  hover?: boolean;
  glass?: boolean;
}
```

### Basic Card

```tsx
<Card title="학습 진행상황">
  <p>Content here</p>
</Card>
```

### With Subtitle

```tsx
<Card 
  title="모델 메트릭"
  subtitle="실시간 업데이트"
>
  <MetricsContent />
</Card>
```

### With Header Action

```tsx
<Card 
  title="최근 프로젝트"
  headerAction={
    <Button variant="ghost" size="sm">
      전체보기
    </Button>
  }
>
  <ProjectList />
</Card>
```

### Hover Effect

```tsx
<Card hover>
  <p>Hover me</p>
</Card>
```

### Glass Effect

```tsx
<Card glass>
  <p>Glassmorphism</p>
</Card>
```

### 구현

```tsx
// components/ui/Card.tsx
import React from 'react';
import { cn } from '@/lib/utils';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
  padding?: 'none' | 'sm' | 'normal' | 'lg';
  hover?: boolean;
  glass?: boolean;
}

const paddingClasses = {
  none: '',
  sm: 'p-4',
  normal: 'p-6',
  lg: 'p-8',
};

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, title, subtitle, headerAction, padding = 'normal', hover, glass, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          "rounded-xl border-2 transition-all duration-200",
          glass 
            ? "bg-white/80 backdrop-blur-xl border-gray-300" 
            : "bg-white border-gray-200",
          hover && "hover:shadow-lg hover:border-gray-300 cursor-pointer",
          !hover && "shadow-md",
          paddingClasses[padding],
          className
        )}
        {...props}
      >
        {(title || headerAction) && (
          <div className="flex items-center justify-between mb-4 pb-4 border-b-2 border-gray-200">
            <div>
              {title && (
                <h3 className="text-lg font-bold text-gray-900">{title}</h3>
              )}
              {subtitle && (
                <p className="text-sm text-gray-600 mt-0.5">{subtitle}</p>
              )}
            </div>
            {headerAction && <div>{headerAction}</div>}
          </div>
        )}
        {children}
      </div>
    );
  }
);

Card.displayName = "Card";
```

---

## Badge

### Props

```typescript
interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'neutral' | 'primary' | 'success' | 'warning' | 'error' | 'gradient';
  size?: 'sm' | 'base' | 'lg';
  dot?: boolean;
}
```

### Variants

```tsx
<Badge variant="neutral">Neutral</Badge>
<Badge variant="primary">Primary</Badge>
<Badge variant="success">Success</Badge>
<Badge variant="warning">Warning</Badge>
<Badge variant="error">Error</Badge>
<Badge variant="gradient">Gradient</Badge>
```

### With Dot

```tsx
<Badge variant="success" dot>
  학습 중
</Badge>
```

### Sizes

```tsx
<Badge size="sm">Small</Badge>
<Badge size="base">Base</Badge>
<Badge size="lg">Large</Badge>
```

### 구현

```tsx
// components/ui/Badge.tsx
import React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  "inline-flex items-center font-semibold rounded-full",
  {
    variants: {
      variant: {
        neutral: "bg-gray-200 text-gray-800",
        primary: "bg-violet-200 text-violet-900",
        success: "bg-emerald-200 text-emerald-900",
        warning: "bg-amber-200 text-amber-900",
        error: "bg-red-200 text-red-900",
        gradient: "bg-gradient-to-r from-violet-600 to-fuchsia-600 text-white",
      },
      size: {
        sm: "px-2 py-0.5 text-xs gap-1",
        base: "px-2.5 py-1 text-xs gap-1.5",
        lg: "px-3 py-1.5 text-sm gap-1.5",
      },
    },
    defaultVariants: {
      variant: "neutral",
      size: "base",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {
  dot?: boolean;
}

export const Badge = React.forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant, size, dot, children, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(badgeVariants({ variant, size }), className)}
        {...props}
      >
        {dot && (
          <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
        )}
        {children}
      </span>
    );
  }
);

Badge.displayName = "Badge";
```

---

## ProgressBar

### Props

```typescript
interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  variant?: 'primary' | 'gradient' | 'success';
  size?: 'sm' | 'base' | 'lg';
}
```

### Basic Usage

```tsx
<ProgressBar value={45} />
```

### With Label

```tsx
<ProgressBar 
  value={45} 
  label="Training Progress"
/>
```

### Gradient Variant

```tsx
<ProgressBar 
  value={78}
  variant="gradient"
  label="Epoch 78/100"
/>
```

### Custom Max

```tsx
<ProgressBar 
  value={78}
  max={100}
  label="Epoch"
/>
```

### Sizes

```tsx
<ProgressBar value={50} size="sm" />
<ProgressBar value={50} size="base" />
<ProgressBar value={50} size="lg" />
```

### 구현

```tsx
// components/ui/ProgressBar.tsx
import React from 'react';
import { cn } from '@/lib/utils';

export interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showPercentage?: boolean;
  variant?: 'primary' | 'gradient' | 'success';
  size?: 'sm' | 'base' | 'lg';
  className?: string;
}

const variantClasses = {
  primary: 'bg-violet-600',
  gradient: 'bg-gradient-to-r from-violet-600 to-fuchsia-600',
  success: 'bg-emerald-600',
};

const sizeClasses = {
  sm: 'h-2',
  base: 'h-3',
  lg: 'h-4',
};

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  showPercentage = true,
  variant = 'primary',
  size = 'base',
  className,
}) => {
  const percentage = Math.round((value / max) * 100);

  return (
    <div className={className}>
      {(label || showPercentage) && (
        <div className="flex justify-between items-center mb-2 text-sm">
          {label && <span className="font-semibold text-gray-900">{label}</span>}
          {showPercentage && <span className="text-gray-600 font-medium">{percentage}%</span>}
        </div>
      )}
      
      <div className={cn(
        "w-full bg-gray-200 rounded-full overflow-hidden shadow-inner",
        sizeClasses[size]
      )}>
        <div
          className={cn(
            "rounded-full transition-all duration-500 ease-out shadow-lg",
            variantClasses[variant],
            sizeClasses[size]
          )}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
```

---

## Modal

### Props

```typescript
interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: React.ReactNode;
}
```

### Basic Usage

```tsx
const [isOpen, setIsOpen] = useState(false);

<Modal 
  isOpen={isOpen}
  onClose={() => setIsOpen(false)}
  title="설정"
>
  <p>Modal content</p>
</Modal>
```

### Sizes

```tsx
<Modal size="sm">Small</Modal>
<Modal size="md">Medium</Modal>
<Modal size="lg">Large</Modal>
<Modal size="xl">Extra Large</Modal>
<Modal size="full">Full Screen</Modal>
```

### 구현

```tsx
// components/ui/Modal.tsx
import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  children: React.ReactNode;
}

const sizeClasses = {
  sm: 'max-w-md',
  md: 'max-w-lg',
  lg: 'max-w-2xl',
  xl: 'max-w-4xl',
  full: 'max-w-full m-4',
};

export const Modal: React.FC<ModalProps> = ({
  isOpen,
  onClose,
  title,
  size = 'md',
  children,
}) => {
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />
      
      {/* Modal */}
      <div
        className={cn(
          "relative bg-white rounded-2xl shadow-2xl w-full animate-scale-in",
          sizeClasses[size]
        )}
      >
        {/* Header */}
        {title && (
          <div className="flex items-center justify-between px-6 py-4 border-b-2 border-gray-200">
            <h2 className="text-xl font-bold text-gray-900">{title}</h2>
            <button
              onClick={onClose}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        )}
        
        {/* Content */}
        <div className="px-6 py-4">
          {children}
        </div>
      </div>
    </div>,
    document.body
  );
};
```

---

## Toast

### Props

```typescript
interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  onClose?: () => void;
}
```

### Usage with Hook

```tsx
// hooks/useToast.ts
import { create } from 'zustand';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
}

interface ToastStore {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
}

export const useToast = create<ToastStore>((set) => ({
  toasts: [],
  addToast: (toast) => {
    const id = Math.random().toString(36);
    set((state) => ({
      toasts: [...state.toasts, { ...toast, id }],
    }));
    
    // Auto remove after 3 seconds
    setTimeout(() => {
      set((state) => ({
        toasts: state.toasts.filter((t) => t.id !== id),
      }));
    }, 3000);
  },
  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),
}));

// Usage
const { addToast } = useToast();

addToast({
  message: '학습이 시작되었습니다',
  type: 'success',
});
```

### 구현

```tsx
// components/ui/Toast.tsx
import React from 'react';
import { CheckCircle, XCircle, AlertCircle, Info, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ToastProps {
  message: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  onClose?: () => void;
}

const icons = {
  success: CheckCircle,
  error: XCircle,
  warning: AlertCircle,
  info: Info,
};

const colorClasses = {
  success: 'bg-emerald-100 border-emerald-300 text-emerald-900',
  error: 'bg-red-100 border-red-300 text-red-900',
  warning: 'bg-amber-100 border-amber-300 text-amber-900',
  info: 'bg-blue-100 border-blue-300 text-blue-900',
};

export const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  onClose,
}) => {
  const Icon = icons[type];

  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3 rounded-lg border-2 shadow-lg animate-slide-up",
        colorClasses[type]
      )}
    >
      <Icon className="w-5 h-5 flex-shrink-0" />
      <p className="flex-1 text-sm font-semibold">{message}</p>
      {onClose && (
        <button
          onClick={onClose}
          className="p-1 hover:bg-black/10 rounded transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

// ToastContainer
export const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useToast();

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <Toast
          key={toast.id}
          message={toast.message}
          type={toast.type}
          onClose={() => removeToast(toast.id)}
        />
      ))}
    </div>
  );
};
```

---

## Dropdown

### Props

```typescript
interface DropdownProps {
  trigger: React.ReactNode;
  items: DropdownItem[];
  align?: 'left' | 'right';
}

interface DropdownItem {
  label: string;
  icon?: React.ReactNode;
  onClick: () => void;
  variant?: 'default' | 'danger';
}
```

### Usage

```tsx
<Dropdown
  trigger={<Button>옵션</Button>}
  items={[
    {
      label: '수정',
      icon: <EditIcon />,
      onClick: () => console.log('edit'),
    },
    {
      label: '삭제',
      icon: <TrashIcon />,
      onClick: () => console.log('delete'),
      variant: 'danger',
    },
  ]}
/>
```

---

## Layout 컴포넌트

### Sidebar

```tsx
// components/layout/Sidebar.tsx
interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}
```

### Header

```tsx
// components/layout/Header.tsx
interface HeaderProps {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}
```

### Main Layout

```tsx
// components/layout/MainLayout.tsx
export const MainLayout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
};
```

---

## 다음 단계

- [디자인 시스템](DESIGN_SYSTEM.md)
- [Storybook 실행](http://localhost:6006)
- [컴포넌트 테스트 작성](../tests/components/)
- [Figma 컴포넌트 라이브러리](https://figma.com/...)
