# Platform Frontend Design System

**Version**: 1.0
**Last Updated**: 2025-01-11
**Status**: Reference Document

---

## Overview

This document defines the design system for the Vision AI Training Platform frontend. It serves as the single source of truth for all visual design decisions, component usage, and UI implementation patterns.

**Based on**: MVP frontend analysis (mvp/frontend/)
**Stack**: Next.js 15 + Tailwind CSS 3.4 + shadcn/ui + TypeScript

---

## Design Principles

### 1. Consistency First
- Use design tokens exclusively (no hardcoded colors)
- Reuse shadcn/ui components whenever possible
- Follow spacing rules (multiples of 4px)
- Maintain pattern consistency across all pages

### 2. Accessibility by Default
- WCAG 2.1 AA compliance
- Keyboard navigation for all interactive elements
- Screen reader support with proper ARIA attributes
- Clear focus indicators

### 3. Mobile-First Responsive
- Design for mobile first, scale up to desktop
- Use 3 primary breakpoints (sm: 640px, md: 768px, lg: 1024px)
- Flexible layouts with Flexbox and Grid
- Touch-friendly tap targets (min 44x44px)

### 4. Performance Optimized
- Next.js Image optimization
- Code splitting with dynamic imports
- Font optimization with next/font
- Lazy loading for heavy components

---

## Color System

### Primary Palette

```typescript
// Design Tokens (CSS Variables)
:root {
  // Background & Foreground
  --background: 0 0% 100%;           // White
  --foreground: 222.2 84% 4.9%;      // Near-black

  // Brand Colors
  --primary: 222.2 47.4% 11.2%;      // Dark navy (primary actions)
  --primary-foreground: 210 40% 98%; // Text on primary

  --secondary: 210 40% 96.1%;        // Light gray (secondary actions)
  --secondary-foreground: 222.2 47.4% 11.2%;

  // Semantic Colors
  --success: 142 76% 36%;            // Green
  --success-foreground: 0 0% 100%;

  --warning: 38 92% 50%;             // Orange
  --warning-foreground: 0 0% 100%;

  --info: 199 89% 48%;               // Blue
  --info-foreground: 0 0% 100%;

  --destructive: 0 84.2% 60.2%;      // Red
  --destructive-foreground: 210 40% 98%;

  // UI Elements
  --muted: 210 40% 96.1%;            // Muted backgrounds
  --muted-foreground: 215.4 16.3% 46.9%;

  --border: 214.3 31.8% 91.4%;       // Borders
  --input: 214.3 31.8% 91.4%;        // Input borders
  --ring: 222.2 84% 4.9%;            // Focus rings

  // Border Radius
  --radius: 0.5rem;                  // 8px
}

.dark {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  // ... (dark mode values)
}
```

### Usage Guidelines

**Primary** - Main actions (Start Training, Submit, Save)
```tsx
<Button variant="default">Start Training</Button>
```

**Secondary** - Supporting actions (Cancel, Back, Skip)
```tsx
<Button variant="secondary">Cancel</Button>
```

**Destructive** - Delete/Remove actions
```tsx
<Button variant="destructive">Delete Job</Button>
```

**Success** - Positive feedback
```tsx
<Alert variant="success">Training completed successfully!</Alert>
```

**Warning** - Caution messages
```tsx
<Alert variant="warning">This action cannot be undone.</Alert>
```

**Error/Destructive** - Error states
```tsx
<Alert variant="destructive">Training failed. Please try again.</Alert>
```

**Info** - Informational messages
```tsx
<Alert variant="info">Your job is queued and will start soon.</Alert>
```

### Color Usage Rules

✅ **DO**:
```tsx
// Use design tokens
<div className="text-primary">Primary text</div>
<div className="bg-success">Success state</div>
<div className="border-destructive">Error border</div>
```

❌ **DON'T**:
```tsx
// Never hardcode colors
<div className="text-blue-600">Blue text</div>
<div className="bg-red-500">Red background</div>
<div style={{ color: '#3b82f6' }}>Inline color</div>
```

---

## Typography

### Font Stack

```typescript
// Font Configuration (layout.tsx)
import { Inter } from "next/font/google";
import localFont from "next/font/local";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const suit = localFont({
  src: "../fonts/SUIT-Variable.woff2",
  variable: "--font-suit",
  display: "swap",
  weight: "100 900",
});

// Tailwind config
fontFamily: {
  sans: [
    "var(--font-suit)",        // Korean (SUIT)
    "var(--font-inter)",       // Latin
    "system-ui",
    "-apple-system",
    "sans-serif",
  ],
  mono: [
    "JetBrains Mono",
    "Menlo",
    "Monaco",
    "Consolas",
    "monospace",
  ],
}
```

### Type Scale

```typescript
// Semantic naming
{
  "heading-1": "2.25rem",   // 36px - Page titles
  "heading-2": "1.875rem",  // 30px - Section titles
  "heading-3": "1.5rem",    // 24px - Card titles
  "heading-4": "1.25rem",   // 20px - Subsections
  "body-lg": "1.125rem",    // 18px - Large body text
  "body": "1rem",           // 16px - Default body text
  "body-sm": "0.875rem",    // 14px - Helper text
  "caption": "0.75rem",     // 12px - Captions, timestamps
}
```

### Usage Examples

```tsx
// Headings
<h1 className="text-4xl font-bold">Page Title</h1>
<h2 className="text-3xl font-semibold">Section Title</h2>
<h3 className="text-2xl font-semibold">Card Title</h3>
<h4 className="text-xl font-medium">Subsection</h4>

// Body text
<p className="text-lg">Lead paragraph</p>
<p className="text-base">Normal body text</p>
<p className="text-sm text-muted-foreground">Helper text</p>
<p className="text-xs text-muted-foreground">Caption or timestamp</p>

// Code
<code className="font-mono text-sm bg-muted px-1.5 py-0.5 rounded">
  model_name
</code>

// Links
<a className="text-primary hover:underline">Read more</a>
```

### Font Weight

```typescript
{
  normal: 400,    // Body text
  medium: 500,    // Emphasized text
  semibold: 600,  // Headings, buttons
  bold: 700,      // Strong emphasis
}
```

---

## Component Guidelines

### Button

**Variants**: `default`, `secondary`, `destructive`, `outline`, `ghost`, `link`
**Sizes**: `default`, `sm`, `lg`, `icon`

```tsx
import { Button } from "@/components/ui/button";

// Primary action
<Button variant="default">Start Training</Button>

// Secondary action
<Button variant="secondary">Cancel</Button>

// Danger action
<Button variant="destructive">Delete</Button>

// Sizes
<Button size="sm">Small</Button>
<Button size="lg">Large</Button>
<Button size="icon"><Icon /></Button>

// With loading state
<Button disabled={isLoading}>
  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
  {isLoading ? "Processing..." : "Submit"}
</Button>

// Icon button with aria-label
<Button size="icon" variant="ghost" aria-label="Delete training job">
  <Trash2 className="h-4 w-4" />
</Button>
```

### Card

```tsx
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";

<Card>
  <CardHeader>
    <CardTitle>Training Progress</CardTitle>
    <CardDescription>Monitor your training in real-time</CardDescription>
  </CardHeader>
  <CardContent>
    {/* Content */}
  </CardContent>
  <CardFooter>
    {/* Actions */}
  </CardFooter>
</Card>
```

### Alert

**Variants**: `default`, `destructive`, `success` (add), `warning` (add), `info` (add)

```tsx
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { AlertCircle, CheckCircle, Info } from "lucide-react";

// Error
<Alert variant="destructive">
  <AlertCircle className="h-4 w-4" />
  <AlertTitle>Error</AlertTitle>
  <AlertDescription>Training failed. Please try again.</AlertDescription>
</Alert>

// Success (requires custom variant)
<Alert variant="success">
  <CheckCircle className="h-4 w-4" />
  <AlertTitle>Success</AlertTitle>
  <AlertDescription>Training completed successfully!</AlertDescription>
</Alert>

// Info
<Alert variant="info">
  <Info className="h-4 w-4" />
  <AlertTitle>Note</AlertTitle>
  <AlertDescription>Your job is queued.</AlertDescription>
</Alert>
```

### Form Components

```tsx
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem } from "@/components/ui/select";

// Complete form field
<div className="space-y-2">
  <Label htmlFor="model-name">
    Model Name
    <span className="text-destructive ml-1" aria-label="required">*</span>
  </Label>
  <Input
    id="model-name"
    type="text"
    value={modelName}
    onChange={handleChange}
    aria-required="true"
    aria-invalid={!!errors.modelName}
    aria-describedby="model-name-description model-name-error"
    placeholder="my-model"
  />
  <p id="model-name-description" className="text-sm text-muted-foreground">
    Choose a unique name for your model
  </p>
  {errors.modelName && (
    <p id="model-name-error" className="text-sm text-destructive" role="alert">
      {errors.modelName}
    </p>
  )}
</div>

// Select dropdown
<div className="space-y-2">
  <Label htmlFor="framework">Framework</Label>
  <Select value={framework} onValueChange={setFramework}>
    <SelectTrigger id="framework">
      <SelectValue placeholder="Select a framework" />
    </SelectTrigger>
    <SelectContent>
      <SelectItem value="timm">TIMM</SelectItem>
      <SelectItem value="ultralytics">Ultralytics YOLO</SelectItem>
      <SelectItem value="huggingface">HuggingFace</SelectItem>
    </SelectContent>
  </Select>
</div>
```

### Progress Indicators

```tsx
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";

// Progress bar
<div className="space-y-2">
  <div className="flex justify-between text-sm">
    <span>Uploading dataset...</span>
    <span>65%</span>
  </div>
  <Progress value={65} />
</div>

// Skeleton loading
<Card>
  <CardHeader>
    <Skeleton className="h-6 w-48" />
    <Skeleton className="h-4 w-64" />
  </CardHeader>
  <CardContent className="space-y-4">
    <Skeleton className="h-32 w-full" />
    <div className="grid grid-cols-2 gap-4">
      <Skeleton className="h-16 w-full" />
      <Skeleton className="h-16 w-full" />
    </div>
  </CardContent>
</Card>
```

---

## Layout Patterns

### Container

```tsx
export function Container({ children, className }: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn(
      "mx-auto max-w-7xl px-4 sm:px-6 lg:px-8",
      className
    )}>
      {children}
    </div>
  );
}
```

### Page Layout

```tsx
<Container>
  <div className="py-8 space-y-8">
    {/* Page Header */}
    <div className="space-y-2">
      <h1 className="text-4xl font-bold">Training Jobs</h1>
      <p className="text-muted-foreground">
        Manage and monitor your training jobs
      </p>
    </div>

    {/* Content */}
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {/* Cards */}
    </div>
  </div>
</Container>
```

### Form Layout

```tsx
<form onSubmit={handleSubmit} className="space-y-6">
  {/* Form sections */}
  <div className="space-y-4">
    <h3 className="text-lg font-semibold">Training Configuration</h3>

    {/* Form fields */}
    <div className="space-y-2">
      <Label>Field Label</Label>
      <Input />
    </div>

    {/* More fields... */}
  </div>

  {/* Actions */}
  <div className="flex gap-2 justify-end">
    <Button type="button" variant="secondary" onClick={onCancel}>
      Cancel
    </Button>
    <Button type="submit" disabled={isSubmitting}>
      {isSubmitting ? "Submitting..." : "Submit"}
    </Button>
  </div>
</form>
```

### Grid Layouts

```tsx
// Responsive grid
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
  {/* Items automatically adjust to breakpoints */}
</div>

// Flexbox alternative
<div className="flex flex-col md:flex-row gap-4">
  <div className="flex-1">{/* Left column */}</div>
  <div className="flex-1">{/* Right column */}</div>
</div>
```

---

## Spacing System

Use multiples of 4px only.

```typescript
// Space between components
{
  "space-y-2": "8px",   // Tight (within components)
  "space-y-4": "16px",  // Normal (form fields)
  "space-y-6": "24px",  // Relaxed (between cards)
  "space-y-8": "32px",  // Loose (between sections)
}

// Padding
{
  "p-4": "16px",  // Small cards
  "p-6": "24px",  // Default cards
  "p-8": "32px",  // Large sections
}

// Margin (use sparingly, prefer space-y)
{
  "mt-4": "16px",
  "mb-6": "24px",
  "my-8": "32px",
}
```

**Usage Guidelines**:
```tsx
// Within a component (8px)
<div className="space-y-2">
  <Label>Label</Label>
  <Input />
  <p className="text-sm">Helper text</p>
</div>

// Form fields (16px)
<form className="space-y-4">
  <div>Field 1</div>
  <div>Field 2</div>
</form>

// Between cards (24px)
<div className="space-y-6">
  <Card>Card 1</Card>
  <Card>Card 2</Card>
</div>

// Between sections (32px)
<div className="space-y-8">
  <section>Section 1</section>
  <section>Section 2</section>
</div>
```

---

## State Patterns

### Loading States

```tsx
// Skeleton (preferred)
import { Skeleton } from "@/components/ui/skeleton";

{isLoading ? (
  <div className="space-y-4">
    <Skeleton className="h-8 w-64" />
    <Skeleton className="h-32 w-full" />
  </div>
) : (
  <div>{/* Actual content */}</div>
)}

// Spinner (for buttons/inline)
import { Loader2 } from "lucide-react";

<Button disabled={isLoading}>
  {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
  {isLoading ? "Loading..." : "Load Data"}
</Button>
```

### Error States

```tsx
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

{error && (
  <Alert variant="destructive">
    <AlertCircle className="h-4 w-4" />
    <AlertTitle>Error</AlertTitle>
    <AlertDescription>{error.message}</AlertDescription>
  </Alert>
)}

// Inline form errors
{errors.fieldName && (
  <p className="text-sm text-destructive" role="alert">
    {errors.fieldName}
  </p>
)}
```

### Empty States

```tsx
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
}: {
  icon?: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      {Icon && (
        <div className="mb-4 rounded-full bg-muted p-3">
          <Icon className="h-6 w-6 text-muted-foreground" />
        </div>
      )}
      <h3 className="text-lg font-semibold">{title}</h3>
      <p className="mt-2 text-sm text-muted-foreground max-w-sm">
        {description}
      </p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

// Usage
<EmptyState
  icon={PackageOpen}
  title="No training jobs yet"
  description="Start your first training job to see it here"
  action={
    <Button onClick={handleStartTraining}>
      <Plus className="mr-2 h-4 w-4" />
      New Training Job
    </Button>
  }
/>
```

### Success States

```tsx
// Success alert
<Alert variant="success">
  <CheckCircle className="h-4 w-4" />
  <AlertTitle>Success!</AlertTitle>
  <AlertDescription>
    Your training job has been created successfully.
  </AlertDescription>
</Alert>

// Toast notification (requires toast setup)
import { useToast } from "@/hooks/use-toast";

const { toast } = useToast();

toast({
  title: "Training started",
  description: "Your job is now running.",
  variant: "success",
});
```

---

## Accessibility Guidelines

### Keyboard Navigation

✅ **Required**:
- All interactive elements accessible via Tab
- Enter/Space activates buttons/links
- Escape closes dialogs/dropdowns
- Arrow keys navigate lists/menus

```tsx
// Custom keyboard-accessible element
<div
  role="button"
  tabIndex={0}
  onClick={handleClick}
  onKeyDown={(e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleClick();
    }
  }}
  className="cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
>
  Clickable element
</div>
```

### ARIA Attributes

```tsx
// Buttons
<Button aria-label="Delete training job">
  <Trash2 className="h-4 w-4" />
</Button>

// Form fields
<Input
  aria-label="Model name"
  aria-required="true"
  aria-invalid={!!errors.modelName}
  aria-describedby="model-name-help"
/>
<p id="model-name-help" className="text-sm">
  Enter a unique name
</p>

// Loading states
<Button disabled={isLoading} aria-busy={isLoading}>
  {isLoading ? "Loading..." : "Submit"}
</Button>

// Error messages
<Alert variant="destructive" role="alert">
  Error message
</Alert>
```

### Focus Indicators

```tsx
// Always visible focus ring
<button className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2">
  Button
</button>

// Skip to main content
<a
  href="#main-content"
  className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary focus:text-primary-foreground focus:rounded-md"
>
  Skip to main content
</a>
```

---

## Responsive Design

### Breakpoints

```typescript
{
  sm: "640px",   // Mobile landscape, small tablets
  md: "768px",   // Tablets
  lg: "1024px",  // Desktop
  xl: "1280px",  // Large desktop
  "2xl": "1536px", // Extra large
}
```

### Mobile-First Approach

```tsx
// ✅ Mobile first (recommended)
<div className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl">
  {/* 2xl on mobile, scales up */}
</div>

// ❌ Desktop first (avoid)
<div className="text-5xl lg:text-4xl md:text-3xl sm:text-2xl">
  {/* Harder to maintain */}
</div>
```

### Responsive Patterns

```tsx
// Typography
<h1 className="text-2xl sm:text-3xl md:text-4xl font-bold">
  Title
</h1>

// Spacing
<div className="space-y-4 sm:space-y-6 lg:space-y-8">
  {/* Adjusts spacing per breakpoint */}
</div>

// Layout
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* 1 column mobile, 2 tablet, 3 desktop */}
</div>

// Visibility
<div className="block md:hidden">Mobile only</div>
<div className="hidden md:block">Desktop only</div>

// Padding
<div className="p-4 sm:p-6 lg:p-8">
  {/* Responsive padding */}
</div>
```

---

## Utility Functions

### cn() - Class Name Utility

```typescript
// lib/utils.ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Usage
import { cn } from "@/lib/utils";

<div className={cn(
  "base-class",
  isActive && "active-class",
  isDisabled && "disabled-class",
  className  // External className override
)} />
```

---

## File Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/                  # shadcn/ui base components
│   │   │   ├── button.tsx
│   │   │   ├── card.tsx
│   │   │   ├── alert.tsx
│   │   │   └── ...
│   │   ├── layout/              # Layout components
│   │   │   ├── Container.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   └── Footer.tsx
│   │   ├── feedback/            # Feedback components
│   │   │   ├── EmptyState.tsx
│   │   │   ├── ErrorBoundary.tsx
│   │   │   └── LoadingScreen.tsx
│   │   └── domain/              # Domain-specific components
│   │       ├── TrainingForm.tsx
│   │       ├── TrainingMonitor.tsx
│   │       ├── ModelCard.tsx
│   │       └── ...
│   ├── lib/
│   │   ├── utils.ts             # cn() and other utilities
│   │   └── api.ts               # API client
│   ├── hooks/
│   │   ├── use-breakpoint.ts
│   │   ├── use-toast.ts
│   │   └── ...
│   ├── app/
│   │   ├── globals.css          # Global styles
│   │   ├── layout.tsx           # Root layout
│   │   └── page.tsx
│   └── fonts/
│       └── SUIT-Variable.woff2
└── tailwind.config.ts
```

---

## Implementation Checklist

### Week 1: Foundation
- [ ] Add missing color tokens (success, warning, info)
- [ ] Install SUIT font
- [ ] Add missing shadcn/ui components (skeleton, toast, dialog)
- [ ] Update globals.css with new tokens

### Week 2: Components
- [ ] Create EmptyState component
- [ ] Create LoadingScreen component
- [ ] Create Container component
- [ ] Update all buttons with loading states

### Week 3: Accessibility
- [ ] Add aria-label to all icon buttons
- [ ] Add aria-describedby to all form fields
- [ ] Add skip-to-main-content link
- [ ] Test keyboard navigation

### Week 4: Responsive
- [ ] Add responsive breakpoints to all layouts
- [ ] Test on mobile devices
- [ ] Add useBreakpoint hook
- [ ] Update typography for mobile

---

## Resources

- [shadcn/ui Documentation](https://ui.shadcn.com/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)
- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [SUIT Font](https://sunn.us/suit/)

---

**Document Status**: ✅ Ready for Implementation
**Next Review**: After Phase 1 Frontend Implementation
