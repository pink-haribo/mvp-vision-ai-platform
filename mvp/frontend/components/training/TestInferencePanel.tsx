'use client'

/**
 * TestInferencePanel Component
 *
 * Interactive inference panel for running predictions on uploaded images.
 * Features image upload, checkpoint selection, task-specific settings, and real-time results.
 */

import { useState, useEffect } from 'react'
import { Upload, Settings, Play, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface TrainingJob {
  id: number
  task_type: string
  framework: string
  model_name: string
  output_dir: string
}

interface TestInferencePanelProps {
  jobId: number
}

interface UploadedImage {
  id: string
  file: File
  preview: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  result?: any
}

interface Checkpoint {
  epoch: number
  path: string
  is_best: boolean
}

export default function TestInferencePanel({ jobId }: TestInferencePanelProps) {
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [images, setImages] = useState<UploadedImage[]>([])
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [loading, setLoading] = useState(true)

  // Inference settings
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [iouThreshold, setIouThreshold] = useState(0.45)
  const [maxDetections, setMaxDetections] = useState(100)
  const [topK, setTopK] = useState(5)

  // Fetch job details
  useEffect(() => {
    const fetchJob = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${jobId}`
        )
        if (response.ok) {
          const data = await response.json()
          setJob(data)
        }
      } catch (error) {
        console.error('Error fetching job:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchJob()
  }, [jobId])

  // Fetch available checkpoints
  useEffect(() => {
    const fetchCheckpoints = async () => {
      try {
        const response = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/training/jobs/${jobId}/checkpoints`
        )
        if (response.ok) {
          const data = await response.json()
          setCheckpoints(data.checkpoints || [])
          // Auto-select best checkpoint
          const best = data.checkpoints?.find((c: Checkpoint) => c.is_best)
          if (best) setSelectedCheckpoint(best)
        }
      } catch (error) {
        console.error('Error fetching checkpoints:', error)
      }
    }
    if (job) fetchCheckpoints()
  }, [job, jobId])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    addImages(files)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'))
    addImages(files)
  }

  const addImages = (files: File[]) => {
    const newImages: UploadedImage[] = files.map(file => ({
      id: `${Date.now()}-${Math.random()}`,
      file,
      preview: URL.createObjectURL(file),
      status: 'pending'
    }))
    setImages(prev => [...prev, ...newImages])
    if (!selectedImageId && newImages.length > 0) {
      setSelectedImageId(newImages[0].id)
    }
  }

  const runInference = async () => {
    if (!selectedCheckpoint || images.length === 0) return

    setIsRunning(true)
    // TODO: Implement actual inference API call
    // For now, just simulate
    setTimeout(() => {
      setIsRunning(false)
    }, 2000)
  }

  const selectedImage = images.find(img => img.id === selectedImageId)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading...</div>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-500">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-gray-400" />
          <p className="text-sm">학습 작업을 찾을 수 없습니다</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Upload and Settings Section */}
      <div className="grid grid-cols-2 gap-6">
        {/* Image Upload */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h3 className="text-sm font-semibold text-gray-900 mb-4">이미지 업로드</h3>

          <div className="space-y-3">
            {/* File input buttons */}
            <div className="flex gap-2">
              <label className={cn(
                'flex-1 px-4 py-2.5',
                'bg-gray-50 hover:bg-gray-100',
                'border border-gray-300 text-gray-700',
                'rounded-lg font-medium text-sm',
                'transition-colors cursor-pointer',
                'flex items-center justify-center gap-2'
              )}>
                <Upload className="w-4 h-4" />
                파일 선택
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>
            </div>

            {/* Upload count */}
            <div className="text-xs text-gray-500">
              업로드된 이미지: <span className="font-medium text-gray-900">{images.length}개</span>
            </div>

            {/* Drop zone */}
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className={cn(
                'border-2 border-dashed border-gray-300',
                'rounded-lg p-8',
                'text-center',
                'hover:border-violet-400 hover:bg-violet-50/50',
                'transition-colors'
              )}
            >
              <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
              <p className="text-sm text-gray-600 mb-1">
                이미지를 드래그 앤 드롭하세요
              </p>
              <p className="text-xs text-gray-500">
                또는 위의 버튼을 클릭하여 선택
              </p>
            </div>
          </div>
        </div>

        {/* Inference Settings */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-4 h-4 text-gray-500" />
            <h3 className="text-sm font-semibold text-gray-900">추론 설정</h3>
          </div>

          <div className="space-y-4">
            {/* Checkpoint Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-2">
                체크포인트 선택
              </label>
              <select
                value={selectedCheckpoint?.epoch || ''}
                onChange={(e) => {
                  const checkpoint = checkpoints.find(c => c.epoch === Number(e.target.value))
                  setSelectedCheckpoint(checkpoint || null)
                }}
                className={cn(
                  'w-full px-3 py-2 text-sm',
                  'border border-gray-300 rounded-lg',
                  'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                  'bg-white'
                )}
              >
                {checkpoints.length === 0 && (
                  <option value="">체크포인트 없음</option>
                )}
                {checkpoints.map(checkpoint => (
                  <option key={checkpoint.epoch} value={checkpoint.epoch}>
                    Epoch {checkpoint.epoch}{checkpoint.is_best ? ' (best)' : ''}
                  </option>
                ))}
              </select>
            </div>

            {/* Task-specific settings */}
            {job.task_type === 'image_classification' && (
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">
                  Top-K Predictions
                </label>
                <input
                  type="number"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  min={1}
                  max={10}
                  className={cn(
                    'w-full px-3 py-2 text-sm',
                    'border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent'
                  )}
                />
              </div>
            )}

            {job.task_type === 'object_detection' && (
              <>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    Confidence Threshold: {confidenceThreshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    IoU Threshold: {iouThreshold.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    value={iouThreshold}
                    onChange={(e) => setIouThreshold(Number(e.target.value))}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-2">
                    Max Detections
                  </label>
                  <input
                    type="number"
                    value={maxDetections}
                    onChange={(e) => setMaxDetections(Number(e.target.value))}
                    min={1}
                    max={300}
                    className={cn(
                      'w-full px-3 py-2 text-sm',
                      'border border-gray-300 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent'
                    )}
                  />
                </div>
              </>
            )}

            {/* Run Inference Button */}
            <button
              onClick={runInference}
              disabled={!selectedCheckpoint || images.length === 0 || isRunning}
              className={cn(
                'w-full px-4 py-2.5',
                'bg-violet-600 hover:bg-violet-700',
                'text-white font-semibold text-sm',
                'rounded-lg shadow-md',
                'transition-all duration-200',
                'disabled:opacity-40 disabled:cursor-not-allowed',
                'flex items-center justify-center gap-2'
              )}
            >
              <Play className="w-4 h-4" />
              {isRunning ? '추론 실행 중...' : '추론 시작'}
            </button>
          </div>
        </div>
      </div>

      {/* Results Section */}
      {images.length > 0 && (
        <div className="grid grid-cols-12 gap-6 h-[600px]">
          {/* Image List */}
          <div className="col-span-2 bg-white rounded-lg border border-gray-200 p-4 overflow-y-auto">
            <h4 className="text-xs font-semibold text-gray-900 mb-3">
              이미지 리스트
            </h4>
            <div className="text-xs text-gray-500 mb-3">
              전체: {images.length}개
            </div>
            <div className="space-y-2">
              {images.map((image) => (
                <div
                  key={image.id}
                  onClick={() => setSelectedImageId(image.id)}
                  className={cn(
                    'cursor-pointer rounded-lg border-2 p-2 transition-all',
                    selectedImageId === image.id
                      ? 'border-violet-600 bg-violet-50'
                      : 'border-gray-200 hover:border-gray-300'
                  )}
                >
                  <img
                    src={image.preview}
                    alt={image.file.name}
                    className="w-full h-16 object-cover rounded mb-1"
                  />
                  <p className="text-xs text-gray-600 truncate">{image.file.name}</p>
                  <div className="flex items-center justify-between mt-1">
                    <span className={cn(
                      'text-xs px-1.5 py-0.5 rounded',
                      image.status === 'completed' && 'bg-green-100 text-green-700',
                      image.status === 'pending' && 'bg-gray-100 text-gray-600',
                      image.status === 'processing' && 'bg-blue-100 text-blue-700',
                      image.status === 'failed' && 'bg-red-100 text-red-700'
                    )}>
                      {image.status === 'completed' && '✓'}
                      {image.status === 'pending' && '⏳'}
                      {image.status === 'processing' && '⚙️'}
                      {image.status === 'failed' && '✗'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Image Viewer */}
          <div className="col-span-6 bg-white rounded-lg border border-gray-200 p-6">
            <h4 className="text-xs font-semibold text-gray-900 mb-3">이미지 뷰어</h4>
            {selectedImage ? (
              <div className="flex items-center justify-center h-[calc(100%-2rem)]">
                <img
                  src={selectedImage.preview}
                  alt={selectedImage.file.name}
                  className="max-w-full max-h-full object-contain rounded"
                />
              </div>
            ) : (
              <div className="flex items-center justify-center h-[calc(100%-2rem)] text-gray-400">
                이미지를 선택하세요
              </div>
            )}
          </div>

          {/* Inference Results */}
          <div className="col-span-4 bg-white rounded-lg border border-gray-200 p-6 overflow-y-auto">
            <h4 className="text-xs font-semibold text-gray-900 mb-3">추론 결과</h4>
            {selectedImage?.result ? (
              <div className="space-y-4 text-sm">
                {/* TODO: Render task-specific results */}
                <p className="text-gray-500">결과가 여기에 표시됩니다</p>
              </div>
            ) : (
              <div className="text-center text-gray-400 py-12">
                <AlertCircle className="w-8 h-8 mx-auto mb-2" />
                <p className="text-xs">추론을 실행하세요</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Empty State */}
      {images.length === 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <Upload className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p className="text-sm text-gray-500 mb-2">
            이미지를 업로드하여 추론을 시작하세요
          </p>
          <p className="text-xs text-gray-400">
            학습된 모델을 사용하여 새로운 이미지에 대한 예측을 수행할 수 있습니다
          </p>
        </div>
      )}
    </div>
  )
}
