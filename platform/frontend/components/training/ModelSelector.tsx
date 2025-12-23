'use client'

import { useState, useEffect } from 'react'
import { Search, Filter, X, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import ModelCard, { ModelInfo } from './ModelCard'
import ModelGuideDrawer from './ModelGuideDrawer'

interface ModelSelectorProps {
  onModelSelect?: (model: ModelInfo) => void
  selectedModel?: ModelInfo | null
  className?: string
}

const FILTER_OPTIONS = {
  framework: [
    { value: 'timm', label: 'timm' },
    { value: 'ultralytics', label: 'Ultralytics' },
    { value: 'huggingface', label: 'HuggingFace' },
    { value: 'mmdet', label: 'MMDetection' },
    { value: 'mmpretrain', label: 'MMPreTrain' },
    { value: 'mmseg', label: 'MMSegmentation' },
    { value: 'mmyolo', label: 'MMYOLO' },
  ],
  taskType: [
    { value: 'detection', label: '객체 탐지' },
    { value: 'segmentation', label: '이미지 분할' },
    { value: 'semantic_segmentation', label: '시맨틱 분할' },
    { value: 'classification', label: '이미지 분류' },
    { value: 'pose', label: '포즈 추정' },
    { value: 'open_vocabulary_detection', label: '오픈 어휘 탐지' },
  ],
}

export default function ModelSelector({
  onModelSelect,
  selectedModel,
  className = '',
}: ModelSelectorProps) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [filteredModels, setFilteredModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [frameworkFilter, setFrameworkFilter] = useState<string>('')
  const [taskTypeFilter, setTaskTypeFilter] = useState<string>('')
  const [showFilters, setShowFilters] = useState(true)

  // Guide drawer
  const [guideOpen, setGuideOpen] = useState(false)
  const [guideFramework, setGuideFramework] = useState('')
  const [guideModelName, setGuideModelName] = useState('')

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
  }, [])

  // Apply filters when models or filters change
  useEffect(() => {
    applyFilters()
  }, [models, searchQuery, frameworkFilter, taskTypeFilter])

  const fetchModels = async () => {
    try {
      setLoading(true)
      setError(null)

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      // Fetch all models including unsupported to show validation status
      const response = await fetch(`${apiUrl}/models/list?supported_only=false`)

      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`)
      }

      const data = await response.json()
      setModels(data)
    } catch (err) {
      console.error('Error fetching models:', err)
      setError(err instanceof Error ? err.message : 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }

  const applyFilters = () => {
    let filtered = [...models]

    // Search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (model) =>
          model.display_name.toLowerCase().includes(query) ||
          model.description.toLowerCase().includes(query) ||
          (model.tags?.some((tag) => tag.toLowerCase().includes(query)) ?? false)
      )
    }

    // Framework filter
    if (frameworkFilter) {
      filtered = filtered.filter((model) => model.framework === frameworkFilter)
    }

    // Task type filter
    if (taskTypeFilter) {
      filtered = filtered.filter((model) => model.task_types.includes(taskTypeFilter))
    }

    setFilteredModels(filtered)
  }

  const clearFilters = () => {
    setSearchQuery('')
    setFrameworkFilter('')
    setTaskTypeFilter('')
  }

  const hasActiveFilters =
    searchQuery || frameworkFilter || taskTypeFilter

  const handleViewGuide = (framework: string, modelName: string) => {
    setGuideFramework(framework)
    setGuideModelName(modelName)
    setGuideOpen(true)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 text-center">
        <div className="text-red-600 mb-2">모델 목록을 불러오는데 실패했습니다</div>
        <div className="text-sm text-gray-600 mb-4">{error}</div>
        <button
          onClick={fetchModels}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          다시 시도
        </button>
      </div>
    )
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="모델 이름, 설명, 태그로 검색..."
          className="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {/* Results Count with Filter Controls */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-600">
          총 <span className="font-bold text-gray-900">{filteredModels.length}</span>개 모델
          {hasActiveFilters && (
            <span className="ml-2 text-gray-500">(전체 {models.length}개 중)</span>
          )}
        </span>
        <div className="flex items-center gap-2">
          {/* Filter Toggle Button */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border transition-colors',
              showFilters || hasActiveFilters
                ? 'bg-blue-50 border-blue-300 text-blue-700'
                : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
            )}
          >
            <Filter className="w-3 h-3" />
            필터
            {hasActiveFilters && (
              <span className="ml-0.5 px-1.5 py-0.5 rounded-full bg-blue-600 text-white text-xs font-bold">
                {[frameworkFilter, taskTypeFilter].filter(Boolean).length}
              </span>
            )}
          </button>
          {/* Clear Filters */}
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium border border-gray-300 text-gray-700 hover:bg-gray-50"
            >
              <X className="w-3 h-3" />
              초기화
            </button>
          )}
        </div>
      </div>

      {/* Filter Options */}
      {showFilters && (
        <div className="p-4 bg-gray-50 rounded-lg border border-gray-200 space-y-2">
          {/* Framework Filter */}
          <div className="flex items-center gap-3">
            <label className="text-xs font-medium text-gray-700 shrink-0">
              프레임워크:
            </label>
            <div className="flex flex-wrap gap-2">
              {FILTER_OPTIONS.framework.map((option) => (
                <button
                  key={option.value}
                  onClick={() =>
                    setFrameworkFilter(
                      frameworkFilter === option.value ? '' : option.value
                    )
                  }
                  className={cn(
                    'px-3 py-1 rounded-full text-xs font-medium transition-colors',
                    frameworkFilter === option.value
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Task Type Filter */}
          <div className="flex items-center gap-3">
            <label className="text-xs font-medium text-gray-700 shrink-0">
              작업 유형:
            </label>
            <div className="flex flex-wrap gap-2">
              {FILTER_OPTIONS.taskType.map((option) => (
                <button
                  key={option.value}
                  onClick={() =>
                    setTaskTypeFilter(
                      taskTypeFilter === option.value ? '' : option.value
                    )
                  }
                  className={cn(
                    'px-3 py-1 rounded-full text-xs font-medium transition-colors',
                    taskTypeFilter === option.value
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Model Grid */}
      {filteredModels.length === 0 ? (
        <div className="py-12 text-center">
          <div className="text-gray-500 mb-2">검색 결과가 없습니다</div>
          <div className="text-sm text-gray-400">
            다른 검색어나 필터를 사용해보세요
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredModels.map((model) => (
            <ModelCard
              key={`${model.framework}-${model.model_name}`}
              model={model}
              selected={
                selectedModel?.framework === model.framework &&
                selectedModel?.model_name === model.model_name
              }
              onSelect={onModelSelect}
              onViewGuide={handleViewGuide}
            />
          ))}
        </div>
      )}

      {/* Model Guide Drawer */}
      <ModelGuideDrawer
        isOpen={guideOpen}
        onClose={() => setGuideOpen(false)}
        framework={guideFramework}
        modelName={guideModelName}
      />
    </div>
  )
}
