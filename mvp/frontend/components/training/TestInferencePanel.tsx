'use client'

/**
 * TestInferencePanel Component
 *
 * Interactive inference panel for running predictions on uploaded images.
 * Features image upload, checkpoint selection, task-specific settings, and real-time results.
 */

import { useState, useEffect, useRef } from 'react'
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
  serverPath?: string  // Server path after upload
  status: 'pending' | 'processing' | 'completed' | 'failed'
  result?: any
  error?: string
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

  // Session ID for image uploads (generated once per component mount)
  const [sessionId] = useState<string>(() => {
    // Generate UUID v4
    return crypto.randomUUID()
  })

  // Inference settings
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [iouThreshold, setIouThreshold] = useState(0.45)
  const [maxDetections, setMaxDetections] = useState(100)
  const [topK, setTopK] = useState(5)

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null)

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

  // Cleanup session on component unmount
  useEffect(() => {
    return () => {
      // Clean up session when leaving the inference panel
      fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/test_inference/inference/session/${sessionId}`,
        { method: 'DELETE' }
      ).catch((error) => {
        // Ignore errors - background cleanup will handle it
        console.log('Session cleanup failed (will be handled by background task):', error)
      })
    }
  }, [sessionId])

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

    try {
      // Run inference on each image
      for (const image of images) {
        // Update status to processing
        setImages(prev => prev.map(img =>
          img.id === image.id ? { ...img, status: 'processing' } : img
        ))

        try {
          // Step 1: Upload image to server if not already uploaded
          let serverPath = image.serverPath
          if (!serverPath) {
            const formData = new FormData()
            formData.append('file', image.file)

            const uploadResponse = await fetch(
              `${process.env.NEXT_PUBLIC_API_URL}/test_inference/inference/upload-image?session_id=${sessionId}`,
              {
                method: 'POST',
                body: formData
              }
            )

            if (!uploadResponse.ok) {
              throw new Error('Failed to upload image')
            }

            const uploadData = await uploadResponse.json()
            serverPath = uploadData.server_path

            // Store server path
            setImages(prev => prev.map(img =>
              img.id === image.id ? { ...img, serverPath } : img
            ))
          }

          // Step 2: Run inference with server path
          const response = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/test_inference/inference/quick?` + new URLSearchParams({
              training_job_id: jobId.toString(),
              checkpoint_path: selectedCheckpoint.path,
              image_path: serverPath,
              confidence_threshold: confidenceThreshold.toString(),
              iou_threshold: iouThreshold.toString(),
              max_detections: maxDetections.toString(),
              top_k: topK.toString()
            }),
            { method: 'POST' }
          )

          if (response.ok) {
            const result = await response.json()
            // Update image with result
            setImages(prev => prev.map(img =>
              img.id === image.id ? { ...img, status: 'completed', result } : img
            ))
          } else {
            const errorData = await response.json().catch(() => ({}))
            throw new Error(errorData.detail || 'Inference failed')
          }
        } catch (error) {
          console.error('Inference error:', error)
          const errorMessage = error instanceof Error ? error.message : 'Unknown error'
          setImages(prev => prev.map(img =>
            img.id === image.id ? { ...img, status: 'failed', error: errorMessage } : img
          ))
        }
      }
    } finally {
      setIsRunning(false)
    }
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
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />

            {/* Upload count */}
            <div className="text-xs text-gray-500">
              업로드된 이미지: <span className="font-medium text-gray-900">{images.length}개</span>
            </div>

            {/* Drop zone - clickable */}
            <div
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className={cn(
                'border-2 border-dashed border-gray-300',
                'rounded-lg p-8',
                'text-center',
                'hover:border-violet-400 hover:bg-violet-50/50',
                'transition-colors cursor-pointer'
              )}
            >
              <Upload className="w-8 h-8 mx-auto mb-2 text-gray-400" />
              <p className="text-sm text-gray-600 mb-1">
                이미지를 드래그 앤 드롭하세요
              </p>
              <p className="text-xs text-gray-500">
                또는 클릭하여 파일 선택
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
                  {image.error && image.status === 'failed' && (
                    <p className="text-xs text-red-600 mt-1 truncate" title={image.error}>
                      {image.error}
                    </p>
                  )}
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
              <div className="space-y-4">
                {/* Performance Metrics */}
                <div className="pb-3 border-b border-gray-200">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-gray-500">추론 시간</span>
                    <span className="font-medium text-gray-900">
                      {selectedImage.result.inference_time_ms?.toFixed(1)}ms
                    </span>
                  </div>
                  {selectedImage.result.preprocessing_time_ms > 0 && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">전처리</span>
                      <span className="font-medium text-gray-700">
                        {selectedImage.result.preprocessing_time_ms?.toFixed(1)}ms
                      </span>
                    </div>
                  )}
                </div>

                {/* Classification Results */}
                {selectedImage.result.task_type === 'image_classification' && (
                  <div>
                    <h5 className="text-xs font-semibold text-gray-900 mb-3">예측 결과</h5>
                    <div className="space-y-2">
                      {selectedImage.result.top5_predictions?.map((pred: any, idx: number) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-medium text-gray-900">
                              {idx === 0 ? '🥇' : idx === 1 ? '🥈' : idx === 2 ? '🥉' : `${idx + 1}.`} {pred.label}
                            </span>
                            <span className="text-gray-600">{(pred.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-1.5">
                            <div
                              className={cn(
                                'h-1.5 rounded-full',
                                idx === 0 ? 'bg-violet-600' : 'bg-gray-400'
                              )}
                              style={{ width: `${pred.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detection Results */}
                {selectedImage.result.task_type === 'object_detection' && (
                  <div>
                    <h5 className="text-xs font-semibold text-gray-900 mb-3">
                      탐지된 객체 ({selectedImage.result.num_detections}개)
                    </h5>
                    <div className="space-y-3">
                      {selectedImage.result.predicted_boxes?.slice(0, 10).map((box: any, idx: number) => (
                        <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-gray-900">
                              📦 Object #{idx + 1}
                            </span>
                            <span className="text-xs px-2 py-0.5 bg-violet-100 text-violet-700 rounded">
                              {(box.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-xs text-gray-600 space-y-1">
                            <div>Class: <span className="font-medium">{box.label}</span></div>
                            <div className="text-xs text-gray-500">
                              BBox: [{box.x1}, {box.y1}, {box.x2}, {box.y2}]
                            </div>
                          </div>
                        </div>
                      ))}
                      {selectedImage.result.num_detections > 10 && (
                        <p className="text-xs text-gray-500 text-center">
                          ...and {selectedImage.result.num_detections - 10} more
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Segmentation Results */}
                {(selectedImage.result.task_type === 'instance_segmentation' ||
                  selectedImage.result.task_type === 'semantic_segmentation') && (
                  <div>
                    <h5 className="text-xs font-semibold text-gray-900 mb-3">
                      분할된 인스턴스 ({selectedImage.result.num_instances}개)
                    </h5>
                    <div className="space-y-3">
                      {selectedImage.result.predicted_boxes?.slice(0, 10).map((box: any, idx: number) => (
                        <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-gray-900">
                              🎨 Instance #{idx + 1}
                            </span>
                            <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded">
                              {(box.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="text-xs text-gray-600">
                            Class: <span className="font-medium">{box.label}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pose Estimation Results */}
                {selectedImage.result.task_type === 'pose_estimation' && (
                  <div>
                    <h5 className="text-xs font-semibold text-gray-900 mb-3">
                      탐지된 사람 ({selectedImage.result.num_persons}명)
                    </h5>
                    <div className="space-y-3">
                      {selectedImage.result.predicted_keypoints?.slice(0, 5).map((person: any, idx: number) => (
                        <div key={idx} className="p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-medium text-gray-900">
                              🧍 Person #{idx + 1}
                            </span>
                            <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded">
                              {person.keypoints?.length || 0} keypoints
                            </span>
                          </div>
                          <div className="text-xs text-gray-600">
                            Detected: {person.num_detected || 0}/17
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
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
