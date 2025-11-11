'use client'

import { cn } from '@/lib/utils/cn'

export type BadgeVariant = 'active' | 'experimental' | 'deprecated' | 'default'

interface BadgeProps {
  variant?: BadgeVariant
  children: React.ReactNode
  className?: string
}

const BADGE_STYLES: Record<BadgeVariant, string> = {
  active: 'bg-green-500 text-white',
  experimental: 'bg-yellow-500 text-white',
  deprecated: 'bg-gray-400 text-white',
  default: 'bg-gray-200 text-gray-700',
}

/**
 * Global Badge Component
 *
 * Used throughout the app for consistent badge styling.
 *
 * Variants:
 * - active: Green badge for active/production-ready models
 * - experimental: Yellow badge for experimental/beta features
 * - deprecated: Gray badge for deprecated/legacy models
 * - default: Default gray badge
 *
 * Example usage:
 * ```tsx
 * <Badge variant="active">Active</Badge>
 * <Badge variant="experimental">Experimental</Badge>
 * <Badge variant="deprecated">Deprecated</Badge>
 * ```
 */
export default function Badge({ variant = 'default', children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-1 rounded-md text-xs font-bold',
        BADGE_STYLES[variant],
        className
      )}
    >
      {children}
    </span>
  )
}
