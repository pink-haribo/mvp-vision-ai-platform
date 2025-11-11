'use client'

import { useState, useEffect } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

// Type definitions matching backend ConfigField
interface ConfigField {
  name: string
  type: 'int' | 'float' | 'str' | 'bool' | 'select' | 'multiselect'
  default: any
  description: string
  required?: boolean
  options?: string[]
  min?: number
  max?: number
  step?: number
  group?: string
  advanced?: boolean
}

interface ConfigSchema {
  fields: ConfigField[]
  presets?: Record<string, Record<string, any>>
}

interface DynamicConfigPanelProps {
  framework: string
  taskType?: string
  config: Record<string, any>
  onChange: (config: Record<string, any>) => void
  className?: string
}

export default function DynamicConfigPanel({
  framework,
  taskType,
  config,
  onChange,
  className
}: DynamicConfigPanelProps) {
  const [schema, setSchema] = useState<ConfigSchema | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(['optimizer', 'scheduler', 'augmentation']))

  // Fetch schema from API
  useEffect(() => {
    const fetchSchema = async () => {
      try {
        setLoading(true)
        const params = new URLSearchParams({ framework })
        if (taskType) params.append('task_type', taskType)

        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/config-schema?${params}`
        )

        if (!response.ok) {
          throw new Error('Failed to fetch configuration schema')
        }

        const data = await response.json()
        // Backend returns { framework, fields, presets } directly
        setSchema({
          fields: data.fields || [],
          presets: data.presets || {}
        })
        setError(null)
      } catch (err) {
        console.error('Error fetching config schema:', err)
        setError(err instanceof Error ? err.message : 'Unknown error')
      } finally {
        setLoading(false)
      }
    }

    if (framework) {
      fetchSchema()
    }
  }, [framework, taskType])

  const updateConfig = (field: string, value: any) => {
    onChange({
      ...config,
      [field]: value
    })
  }

  const applyPreset = (presetName: string) => {
    if (!schema?.presets || !schema.presets[presetName]) return

    const presetConfig = schema.presets[presetName]
    onChange({
      ...config,
      ...presetConfig
    })
  }

  const toggleGroup = (group: string) => {
    const newExpanded = new Set(expandedGroups)
    if (newExpanded.has(group)) {
      newExpanded.delete(group)
    } else {
      newExpanded.add(group)
    }
    setExpandedGroups(newExpanded)
  }

  const renderField = (field: ConfigField) => {
    // Skip advanced fields if not shown
    if (field.advanced && !showAdvanced) return null

    const value = config[field.name] ?? field.default

    switch (field.type) {
      case 'bool':
        return (
          <div key={field.name} className="flex items-center p-3 bg-gray-700/30 rounded-lg">
            <input
              type="checkbox"
              id={field.name}
              checked={value}
              onChange={(e) => updateConfig(field.name, e.target.checked)}
              className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
            />
            <label htmlFor={field.name} className="ml-2 text-sm text-gray-300">
              {field.description}
              {field.advanced && <span className="ml-2 text-xs text-amber-400">(Advanced)</span>}
            </label>
          </div>
        )

      case 'int':
      case 'float':
        return (
          <div key={field.name} className="p-3 bg-gray-700/30 rounded-lg">
            <label htmlFor={field.name} className="block text-sm text-gray-300 mb-1">
              {field.description}
              {field.advanced && <span className="ml-2 text-xs text-amber-400">(Advanced)</span>}
            </label>
            <input
              type="number"
              id={field.name}
              value={value}
              onChange={(e) => updateConfig(field.name, field.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
              min={field.min}
              max={field.max}
              step={field.step}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-white focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
            {(field.min !== undefined || field.max !== undefined) && (
              <p className="mt-1 text-xs text-gray-500">
                Range: {field.min ?? '−∞'} to {field.max ?? '∞'}
              </p>
            )}
          </div>
        )

      case 'select':
        return (
          <div key={field.name} className="p-3 bg-gray-700/30 rounded-lg">
            <label htmlFor={field.name} className="block text-sm text-gray-300 mb-1">
              {field.description}
              {field.advanced && <span className="ml-2 text-xs text-amber-400">(Advanced)</span>}
            </label>
            <select
              id={field.name}
              value={value}
              onChange={(e) => updateConfig(field.name, e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-white focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              {field.options?.map(option => (
                <option key={option} value={option}>{option}</option>
              ))}
            </select>
          </div>
        )

      case 'str':
        return (
          <div key={field.name} className="p-3 bg-gray-700/30 rounded-lg">
            <label htmlFor={field.name} className="block text-sm text-gray-300 mb-1">
              {field.description}
              {field.advanced && <span className="ml-2 text-xs text-amber-400">(Advanced)</span>}
            </label>
            <input
              type="text"
              id={field.name}
              value={value}
              onChange={(e) => updateConfig(field.name, e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-sm text-white focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
        )

      default:
        return null
    }
  }

  if (loading) {
    return (
      <div className={cn("bg-gray-800 rounded-lg p-6", className)}>
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-500"></div>
          <span className="ml-3 text-gray-400">Loading configuration options...</span>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={cn("bg-gray-800 rounded-lg p-6", className)}>
        <div className="text-red-400">
          <p className="font-semibold">Error loading configuration</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    )
  }

  if (!schema) {
    return null
  }

  // Group fields by group name
  const groupedFields = schema.fields.reduce((acc, field) => {
    const group = field.group || 'general'
    if (!acc[group]) acc[group] = []
    acc[group].push(field)
    return acc
  }, {} as Record<string, ConfigField[]>)

  // Count how many advanced fields are available
  const advancedCount = schema.fields.filter(f => f.advanced).length

  return (
    <div className={cn("bg-gray-800 rounded-lg p-4 space-y-4", className)}>
      {/* Header with Presets */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Advanced Configuration</h3>
        {schema.presets && Object.keys(schema.presets).length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">Preset:</span>
            {Object.keys(schema.presets).map(presetName => (
              <button
                key={presetName}
                onClick={() => applyPreset(presetName)}
                className="px-3 py-1 text-xs font-medium text-gray-300 bg-gray-700 hover:bg-gray-600 rounded transition-colors capitalize"
              >
                {presetName}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Advanced Toggle */}
      {advancedCount > 0 && (
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-4 py-2 text-sm text-left text-gray-300 bg-gray-700/50 hover:bg-gray-700 rounded transition-colors flex items-center justify-between"
        >
          <span>
            {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            <span className="ml-2 text-xs text-gray-500">({advancedCount} options)</span>
          </span>
          {showAdvanced ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      )}

      {/* Grouped Fields */}
      <div className="space-y-3">
        {Object.entries(groupedFields).map(([group, fields]) => {
          const isExpanded = expandedGroups.has(group)

          return (
            <div key={group} className="bg-gray-700/30 rounded-lg">
              <button
                onClick={() => toggleGroup(group)}
                className="w-full px-4 py-3 text-left flex items-center justify-between hover:bg-gray-700/50 transition-colors rounded-lg"
              >
                <div className="flex items-center gap-2">
                  {isExpanded ? (
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                  ) : (
                    <ChevronRight className="w-5 h-5 text-gray-400" />
                  )}
                  <h4 className="text-md font-semibold text-white capitalize">{group}</h4>
                </div>
                <span className="text-sm text-gray-400">
                  {fields.length} option{fields.length !== 1 ? 's' : ''}
                </span>
              </button>

              {isExpanded && (
                <div className="px-4 pb-4 space-y-2">
                  {fields.map(field => renderField(field))}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
