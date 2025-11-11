'use client'

import { useState } from 'react'
import { ArrowLeftIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

interface CreateProjectFormProps {
  onCancel: () => void
  onProjectCreated: (projectId: number) => void
}

export default function CreateProjectForm({
  onCancel,
  onProjectCreated,
}: CreateProjectFormProps) {
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!name.trim()) {
      setError('프로젝트 이름을 입력해주세요')
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      const token = localStorage.getItem('access_token')
      if (!token) {
        throw new Error('로그인이 필요합니다. 다시 로그인해주세요.')
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim() || null,
          task_type: null,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))

        // Handle authentication error
        if (response.status === 401) {
          throw new Error('인증이 만료되었습니다. 다시 로그인해주세요.')
        }

        // Handle duplicate project name (400 error)
        if (response.status === 400 && errorData.detail?.includes('already exists')) {
          throw new Error('이미 존재하는 프로젝트 이름입니다. 다른 이름을 사용해주세요.')
        }

        throw new Error(errorData.detail || '프로젝트 생성에 실패했습니다')
      }

      const project = await response.json()
      console.log('Project created:', project)

      // Notify parent and show project detail
      onProjectCreated(project.id)
    } catch (err) {
      console.error('Error creating project:', err)
      setError(err instanceof Error ? err.message : '프로젝트 생성 중 오류가 발생했습니다')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center space-x-4">
          <button
            onClick={onCancel}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeftIcon className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">새 프로젝트 만들기</h2>
            <p className="text-sm text-gray-600 mt-1">
              실험을 그룹화할 프로젝트를 생성하세요
            </p>
          </div>
        </div>
      </div>

      {/* Form */}
      <div className="flex-1 overflow-y-auto p-6">
        <form onSubmit={handleSubmit} className="max-w-2xl mx-auto space-y-6">
          {/* Error Message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {/* Project Name */}
          <div>
            <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
              프로젝트 이름 <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="예: 이미지 분류 프로젝트"
              className={cn(
                'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                'text-sm'
              )}
              maxLength={100}
              disabled={isSubmitting}
            />
            <p className="text-xs text-gray-500 mt-1">
              {name.length}/100 자
            </p>
          </div>

          {/* Description */}
          <div>
            <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-2">
              설명 (선택)
            </label>
            <textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="예: ResNet 모델을 활용한 이미지 분류 실험 프로젝트"
              rows={3}
              className={cn(
                'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                'text-sm resize-none'
              )}
              maxLength={500}
              disabled={isSubmitting}
            />
            <p className="text-xs text-gray-500 mt-1">
              {description.length}/500 자
            </p>
          </div>

          {/* Info Box */}
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-900 mb-2">
              프로젝트란?
            </h3>
            <ul className="text-xs text-blue-800 space-y-1">
              <li>• 관련된 실험들을 그룹화하여 관리할 수 있습니다</li>
              <li>• 프로젝트별로 실험 결과를 비교하고 분석할 수 있습니다</li>
              <li>• MLflow Experiment와 자동으로 연결됩니다</li>
              <li>• 프로젝트 없이도 실험을 진행할 수 있습니다</li>
            </ul>
          </div>

          {/* Buttons */}
          <div className="flex gap-3 pt-4">
            <button
              type="button"
              onClick={onCancel}
              disabled={isSubmitting}
              className={cn(
                'flex-1 px-4 py-2.5',
                'border border-gray-300 text-gray-700',
                'rounded-lg font-medium',
                'hover:bg-gray-50',
                'transition-colors',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              취소
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !name.trim()}
              className={cn(
                'flex-1 px-4 py-2.5',
                'bg-violet-600 text-white',
                'rounded-lg font-medium',
                'hover:bg-violet-700',
                'transition-colors',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              {isSubmitting ? '생성 중...' : '프로젝트 생성'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
