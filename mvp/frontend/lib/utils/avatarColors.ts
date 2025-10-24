/**
 * Avatar badge color utilities
 *
 * Provides consistent solid colors for user avatar badges based on the badge_color field.
 * Colors are assigned randomly during user registration and remain consistent for each user.
 */

export type AvatarColor =
  | 'red' | 'orange' | 'green' | 'emerald'
  | 'teal' | 'cyan' | 'sky' | 'blue'
  | 'indigo' | 'violet' | 'purple' | 'fuchsia' | 'pink'

/**
 * Get inline style object for avatar badge background
 * Returns RGB color for reliable rendering
 */
export function getAvatarColorStyle(color: string | null | undefined): { backgroundColor: string } {
  const colors: Record<string, string> = {
    red: '#dc2626',       // red-600
    orange: '#ea580c',    // orange-600
    green: '#16a34a',     // green-600
    emerald: '#059669',   // emerald-600
    teal: '#0d9488',      // teal-600
    cyan: '#0891b2',      // cyan-600
    sky: '#0284c7',       // sky-600
    blue: '#2563eb',      // blue-600
    indigo: '#4f46e5',    // indigo-600
    violet: '#7c3aed',    // violet-600
    purple: '#9333ea',    // purple-600
    fuchsia: '#c026d3',   // fuchsia-600
    pink: '#db2777',      // pink-600
  }

  const bgColor = colors[color || ''] || colors.violet
  return { backgroundColor: bgColor }
}

/**
 * @deprecated Use getAvatarColorStyle instead
 */
export function getAvatarColor(color: string | null | undefined): string {
  // This function is kept for compatibility but won't work with Tailwind JIT
  return 'bg-violet-600'
}

/**
 * @deprecated Use getAvatarColorStyle instead
 */
export function getAvatarGradient(color: string | null | undefined): string {
  return getAvatarColor(color)
}

/**
 * Get Tailwind ring color classes for avatar badge hover effect
 */
export function getAvatarRingColor(color: string | null | undefined): string {
  if (!color) {
    return 'hover:ring-violet-400'
  }

  const ringColors: Record<string, string> = {
    red: 'hover:ring-red-400',
    orange: 'hover:ring-orange-400',
    green: 'hover:ring-green-400',
    emerald: 'hover:ring-emerald-400',
    teal: 'hover:ring-teal-400',
    cyan: 'hover:ring-cyan-400',
    sky: 'hover:ring-sky-400',
    blue: 'hover:ring-blue-400',
    indigo: 'hover:ring-indigo-400',
    violet: 'hover:ring-violet-400',
    purple: 'hover:ring-purple-400',
    fuchsia: 'hover:ring-fuchsia-400',
    pink: 'hover:ring-pink-400',
  }

  return ringColors[color] || 'hover:ring-violet-400'
}

/**
 * Get complete avatar badge classes including background color and ring
 */
export function getAvatarBadgeClasses(color: string | null | undefined): string {
  const bgColor = getAvatarColor(color)
  const ring = getAvatarRingColor(color)

  return `${bgColor} ${ring}`
}
