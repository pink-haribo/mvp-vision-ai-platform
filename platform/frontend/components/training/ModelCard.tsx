'use client'

import { useState } from 'react'
import { ChevronRight, Sparkles, Target } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import Badge from '@/components/ui/Badge'

export interface ModelInfo {
  framework: string
  model_name: string
  display_name: string
  description: string
  params: string
  input_size: number
  task_types: string[]  // Changed from task_type to task_types (array)
  pretrained_available: boolean
  recommended_batch_size: number
  recommended_lr: number
  tags: string[]
  status?: 'active' | 'experimental' | 'deprecated'
  benchmark?: Record<string, any>
  special_features?: {
    type: string
    capabilities?: string[]
    [key: string]: any
  }
}

interface ModelCardProps {
  model: ModelInfo
  onViewGuide?: (framework: string, modelName: string) => void
  onSelect?: (model: ModelInfo) => void
  selected?: boolean
  className?: string
}

const TASK_TYPE_LABELS: Record<string, string> = {
  'image_classification': '이미지 분류',
  'object_detection': '객체 탐지',
  'instance_segmentation': '인스턴스 분할',
  'pose_estimation': '포즈 추정',
  'zero_shot_detection': '제로샷 탐지',
}

const FRAMEWORK_LABELS: Record<string, string> = {
  'timm': 'timm',
  'ultralytics': 'Ultralytics',
}

export default function ModelCard({
  model,
  onViewGuide,
  onSelect,
  selected = false,
  className = '',
}: ModelCardProps) {
  const [isHovered, setIsHovered] = useState(false)

  const taskTypeLabel = TASK_TYPE_LABELS[model.task_types[0]] || model.task_types[0]
  const frameworkLabel = FRAMEWORK_LABELS[model.framework] || model.framework
  const status = model.status || 'active' // Default to 'active' if not specified

  const hasSpecialFeatures = !!model.special_features

  // Extract key benchmark metric
  const getKeyMetric = () => {
    if (!model.benchmark) return null

    const { benchmark } = model

    // Classification
    if (benchmark.imagenet_top1) {
      return { label: 'Top-1 Acc', value: `${benchmark.imagenet_top1}%` }
    }

    // Detection
    if (benchmark.coco_map50_95) {
      return { label: 'mAP50-95', value: `${benchmark.coco_map50_95}%` }
    }

    // Zero-shot detection
    if (benchmark.lvis_map) {
      return { label: 'LVIS mAP', value: `${benchmark.lvis_map}%` }
    }

    return null
  }

  const keyMetric = getKeyMetric()

  const handleCardClick = () => {
    if (onSelect) {
      onSelect(model)
    }
  }

  return (
    <div
      className={cn(
        'relative rounded-lg border-2 transition-all duration-200 cursor-pointer',
        selected
          ? 'border-blue-500 bg-blue-50 shadow-lg'
          : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md',
        className
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleCardClick}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-start justify-between mb-2">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <h3 className="text-lg font-bold text-gray-900">
                {model.display_name}
              </h3>
              {hasSpecialFeatures && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-700">
                  <Sparkles className="w-3 h-3" />
                  특별 기능
                </span>
              )}
            </div>
            <p className="text-sm text-gray-600 line-clamp-2">
              {model.description}
            </p>
          </div>

          {/* Status Badge */}
          <Badge variant={status} className="ml-3 shrink-0">
            {status === 'active' && 'Active'}
            {status === 'experimental' && 'Experimental'}
            {status === 'deprecated' && 'Deprecated'}
          </Badge>
        </div>

        {/* Badges */}
        <div className="flex flex-wrap gap-2 mt-3">
          <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-700">
            {frameworkLabel}
          </span>
          <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-700">
            {taskTypeLabel}
          </span>
        </div>
      </div>

      {/* Body */}
      <div className="p-4">
        {/* Key Metrics */}
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Parameters</div>
            <div className="text-sm font-bold text-gray-900">{model.params}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-gray-500 mb-1">Input Size</div>
            <div className="text-sm font-bold text-gray-900">{model.input_size}px</div>
          </div>
          {keyMetric && (
            <div className="text-center">
              <div className="text-xs text-gray-500 mb-1">{keyMetric.label}</div>
              <div className="text-sm font-bold text-gray-900">{keyMetric.value}</div>
            </div>
          )}
        </div>

        {/* Special Features Preview */}
        {hasSpecialFeatures && model.special_features && (
          <div className="mb-4 p-3 rounded-lg bg-purple-50 border border-purple-100">
            <div className="flex items-start gap-2">
              <Target className="w-4 h-4 text-purple-600 mt-0.5 shrink-0" />
              <div>
                <div className="text-xs font-semibold text-purple-900 mb-1">
                  {model.special_features.type === 'open_vocabulary' && 'Open-Vocabulary Detection'}
                </div>
                {model.special_features.capabilities && (
                  <div className="text-xs text-purple-700">
                    {model.special_features.capabilities[0]}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* View Guide Button */}
        <button
          onClick={(e) => {
            e.stopPropagation()
            if (onViewGuide) {
              onViewGuide(model.framework, model.model_name)
            }
          }}
          className={cn(
            'w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors',
            isHovered || selected
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          )}
        >
          가이드 보기
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      {/* Selected Indicator */}
      {selected && (
        <div className="absolute top-2 right-2 w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        </div>
      )}
    </div>
  )
}
