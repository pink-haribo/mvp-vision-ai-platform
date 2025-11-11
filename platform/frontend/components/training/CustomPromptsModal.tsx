'use client'

import { useState } from 'react'
import { X, Plus, Trash2, Sparkles, AlertCircle, CheckCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

interface CustomPromptsModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: (prompts: string[]) => void
  initialPrompts?: string[]
  modelName?: string
}

const EXAMPLE_PROMPTS = [
  'person',
  'car',
  'bicycle',
  'dog',
  'cat',
  'traffic light',
  'stop sign',
  'fire hydrant',
  'damaged packaging',
  'defective product',
]

const PROMPT_TIPS = [
  '구체적으로 작성하세요: "빨간 사과"가 "사과"보다 효과적입니다',
  '속성을 추가하세요: "손상된", "익은", "빈티지" 등',
  '객체 + 상태 조합: "마스크를 쓴 사람"',
  '모호함을 피하세요: 색상, 크기, 상태를 명시하세요',
]

export default function CustomPromptsModal({
  isOpen,
  onClose,
  onConfirm,
  initialPrompts = [],
  modelName = 'YOLO-World',
}: CustomPromptsModalProps) {
  const [prompts, setPrompts] = useState<string[]>(
    initialPrompts.length > 0 ? initialPrompts : ['']
  )
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const handleAddPrompt = () => {
    setPrompts([...prompts, ''])
    setError(null)
  }

  const handleRemovePrompt = (index: number) => {
    if (prompts.length === 1) {
      setError('최소 1개의 프롬프트가 필요합니다')
      return
    }
    const newPrompts = prompts.filter((_, i) => i !== index)
    setPrompts(newPrompts)
    setError(null)
  }

  const handlePromptChange = (index: number, value: string) => {
    const newPrompts = [...prompts]
    newPrompts[index] = value
    setPrompts(newPrompts)
    setError(null)
  }

  const handleAddExample = (example: string) => {
    // Add to the last empty prompt, or create a new one
    const lastEmptyIndex = prompts.findIndex((p) => p.trim() === '')
    if (lastEmptyIndex >= 0) {
      const newPrompts = [...prompts]
      newPrompts[lastEmptyIndex] = example
      setPrompts(newPrompts)
    } else {
      setPrompts([...prompts, example])
    }
    setError(null)
  }

  const handleConfirm = () => {
    // Filter out empty prompts
    const validPrompts = prompts.filter((p) => p.trim() !== '')

    if (validPrompts.length === 0) {
      setError('최소 1개의 프롬프트를 입력하세요')
      return
    }

    // Check for duplicates
    const uniquePrompts = [...new Set(validPrompts)]
    if (uniquePrompts.length !== validPrompts.length) {
      setError('중복된 프롬프트가 있습니다')
      return
    }

    onConfirm(uniquePrompts)
    onClose()
  }

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose()
    }
  }

  const validPromptsCount = prompts.filter((p) => p.trim() !== '').length

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 bg-purple-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-purple-100 flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-purple-600" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900">
                  텍스트 프롬프트 설정
                </h2>
                <p className="text-sm text-gray-600">
                  {modelName}에서 탐지할 객체를 자연어로 정의하세요
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 hover:bg-purple-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Info Banner */}
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg flex gap-3">
            <AlertCircle className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
            <div className="text-sm text-blue-800">
              <p className="font-semibold mb-1">Open-Vocabulary Detection이란?</p>
              <p>
                학습 없이 텍스트 프롬프트만으로 새로운 객체를 탐지할 수 있습니다.
                전통적인 YOLO는 80개 고정 클래스만 탐지하지만, YOLO-World는 무제한 클래스를 지원합니다.
              </p>
            </div>
          </div>

          {/* Prompt Tips */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">
              💡 효과적인 프롬프트 작성 팁
            </h3>
            <div className="space-y-2">
              {PROMPT_TIPS.map((tip, index) => (
                <div key={index} className="flex gap-2 text-sm text-gray-700">
                  <CheckCircle className="w-4 h-4 text-green-600 shrink-0 mt-0.5" />
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Prompt List */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-900">
                커스텀 클래스 ({validPromptsCount}개)
              </h3>
              <button
                onClick={handleAddPrompt}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-blue-700 bg-blue-50 rounded-md hover:bg-blue-100 transition-colors"
              >
                <Plus className="w-4 h-4" />
                추가
              </button>
            </div>

            <div className="space-y-3">
              {prompts.map((prompt, index) => (
                <div key={index} className="flex gap-2">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={prompt}
                      onChange={(e) => handlePromptChange(index, e.target.value)}
                      placeholder={`클래스 ${index + 1} (예: person, car, damaged box)`}
                      className={cn(
                        'w-full px-4 py-2.5 border rounded-lg',
                        'focus:outline-none focus:ring-2 focus:ring-purple-500',
                        prompt.trim() ? 'border-gray-300' : 'border-gray-200 bg-gray-50'
                      )}
                    />
                    {prompt.trim() && (
                      <div className="absolute right-3 top-1/2 -translate-y-1/2">
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      </div>
                    )}
                  </div>
                  <button
                    onClick={() => handleRemovePrompt(index)}
                    className={cn(
                      'p-2.5 rounded-lg transition-colors',
                      prompts.length === 1
                        ? 'text-gray-300 cursor-not-allowed'
                        : 'text-gray-600 hover:bg-red-50 hover:text-red-600'
                    )}
                    disabled={prompts.length === 1}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Example Prompts */}
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">
              📚 예시 클래스 (클릭하여 추가)
            </h3>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_PROMPTS.map((example) => (
                <button
                  key={example}
                  onClick={() => handleAddExample(example)}
                  className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-purple-100 hover:text-purple-700 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-between">
          <div className="text-sm text-gray-600">
            {validPromptsCount > 0 ? (
              <span>
                <span className="font-semibold text-purple-700">{validPromptsCount}개</span> 클래스가 설정됩니다
              </span>
            ) : (
              <span className="text-gray-500">최소 1개의 프롬프트를 입력하세요</span>
            )}
          </div>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
            >
              취소
            </button>
            <button
              onClick={handleConfirm}
              disabled={validPromptsCount === 0}
              className={cn(
                'px-4 py-2 text-sm font-medium rounded-lg transition-colors',
                validPromptsCount > 0
                  ? 'bg-purple-600 text-white hover:bg-purple-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              )}
            >
              확인
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
