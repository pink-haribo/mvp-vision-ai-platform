'use client'

import { useState, useEffect } from 'react'
import { FolderIcon, PlusIcon, ChevronRightIcon } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface Project {
  id: number
  name: string
  description: string | null
  task_type: string | null
  created_at: string
  updated_at: string
  experiment_count: number
}

interface ProjectListProps {
  onProjectSelect?: (projectId: number) => void
}

export default function ProjectList({ onProjectSelect }: ProjectListProps) {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null)

  useEffect(() => {
    fetchProjects()
  }, [])

  const fetchProjects = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/projects`)
      if (response.ok) {
        const data = await response.json()
        setProjects(data)
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleProjectClick = (projectId: number) => {
    setSelectedProjectId(projectId)
    if (onProjectSelect) {
      onProjectSelect(projectId)
    }
  }

  const getTaskTypeLabel = (taskType: string | null) => {
    const labels: Record<string, string> = {
      'image_classification': '이미지 분류',
      'object_detection': '객체 탐지',
      'instance_segmentation': '인스턴스 분할',
      'pose_estimation': '자세 추정',
    }
    return taskType ? labels[taskType] || taskType : '전체'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">프로젝트 로딩 중...</div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">프로젝트</h2>
          <button
            onClick={() => {/* TODO: Open create project modal */}}
            className="p-2 text-violet-600 hover:bg-violet-50 rounded-lg transition-colors"
            title="새 프로젝트"
          >
            <PlusIcon className="w-5 h-5" />
          </button>
        </div>
        <p className="text-sm text-gray-600 mt-1">
          {projects.length}개의 프로젝트
        </p>
      </div>

      {/* Project List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {projects.map((project) => (
          <div
            key={project.id}
            onClick={() => handleProjectClick(project.id)}
            className={cn(
              'p-4 rounded-lg border cursor-pointer transition-all',
              'hover:shadow-md hover:border-violet-300',
              selectedProjectId === project.id
                ? 'bg-violet-50 border-violet-400 shadow-sm'
                : 'bg-white border-gray-200'
            )}
          >
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-3 flex-1">
                <div className={cn(
                  'p-2 rounded-lg',
                  selectedProjectId === project.id
                    ? 'bg-violet-100'
                    : 'bg-gray-100'
                )}>
                  <FolderIcon className={cn(
                    'w-5 h-5',
                    selectedProjectId === project.id
                      ? 'text-violet-600'
                      : 'text-gray-600'
                  )} />
                </div>

                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-gray-900 truncate">
                    {project.name}
                  </h3>
                  {project.description && (
                    <p className="text-sm text-gray-600 mt-1 line-clamp-2">
                      {project.description}
                    </p>
                  )}
                  <div className="flex items-center gap-3 mt-2">
                    {project.task_type && (
                      <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-blue-100 text-blue-700">
                        {getTaskTypeLabel(project.task_type)}
                      </span>
                    )}
                    <span className="text-xs text-gray-500">
                      실험 {project.experiment_count}개
                    </span>
                  </div>
                </div>
              </div>

              <ChevronRightIcon className="w-5 h-5 text-gray-400 flex-shrink-0 ml-2" />
            </div>
          </div>
        ))}

        {projects.length === 0 && (
          <div className="text-center py-12">
            <FolderIcon className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-gray-500 text-sm">프로젝트가 없습니다</p>
            <p className="text-gray-400 text-xs mt-1">
              새 프로젝트를 만들거나 학습을 시작하세요
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
