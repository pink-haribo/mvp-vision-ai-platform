'use client'

import { useState, useRef, useCallback } from 'react'
import { Upload, X, Loader2, PlayCircle, Image as ImageIcon, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface InferenceTestPanelProps {
  deploymentId: number
  apiKey: string
  endpointUrl: string
  taskType?: string
}

interface Detection {
  class_id: number
  class_name?: string
  confidence: number
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

interface InferenceResult {
  deployment_id: number
  task_type: string
  inference_time_ms: number
  detections?: Detection[]
  classification_result?: {
    class_id: number
    class_name: string
    confidence: number
  }
}

export default function InferenceTestPanel({ deploymentId, apiKey, endpointUrl, taskType = 'detection' }: InferenceTestPanelProps) {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25)
  const [iouThreshold, setIouThreshold] = useState(0.45)
  const [maxDetections, setMaxDetections] = useState(100)
  const [isInferring, setIsInferring] = useState(false)
  const [result, setResult] = useState<InferenceResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleFileSelect = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file')
      return
    }

    setSelectedImage(file)
    setError(null)
    setResult(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileSelect(file)
    }
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const drawDetections = (imageUrl: string, detections: Detection[]) => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width
      canvas.height = img.height

      // Draw image
      ctx.drawImage(img, 0, 0)

      // Draw detections
      detections.forEach((det, idx) => {
        const { x1, y1, x2, y2 } = det.bbox
        const width = x2 - x1
        const height = y2 - y1

        // Use different colors for different classes
        const colors = ['#EF4444', '#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
        const color = colors[det.class_id % colors.length]

        // Draw bounding box
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(x1, y1, width, height)

        // Draw label background
        const label = `${det.class_name || `Class ${det.class_id}`} ${(det.confidence * 100).toFixed(1)}%`
        ctx.font = '14px Inter, sans-serif'
        const textMetrics = ctx.measureText(label)
        const textHeight = 20

        ctx.fillStyle = color
        ctx.fillRect(x1, y1 - textHeight - 4, textMetrics.width + 8, textHeight + 4)

        // Draw label text
        ctx.fillStyle = '#FFFFFF'
        ctx.fillText(label, x1 + 4, y1 - 8)
      })
    }
    img.src = imageUrl
  }

  const handleInference = async () => {
    if (!selectedImage) {
      setError('Please select an image first')
      return
    }

    try {
      setIsInferring(true)
      setError(null)
      setResult(null)

      // Convert image to base64
      const reader = new FileReader()
      reader.onload = async (e) => {
        try {
          const base64Image = (e.target?.result as string).split(',')[1]

          const requestBody = {
            image: base64Image,
            confidence_threshold: confidenceThreshold,
            iou_threshold: iouThreshold,
            max_detections: maxDetections
          }

          const response = await fetch(endpointUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`
            },
            body: JSON.stringify(requestBody)
          })

          if (!response.ok) {
            const errorData = await response.json()
            throw new Error(errorData.detail || 'Inference failed')
          }

          const data: InferenceResult = await response.json()
          setResult(data)

          // Draw detections on canvas if available
          if (data.detections && data.detections.length > 0 && imagePreview) {
            drawDetections(imagePreview, data.detections)
          }
        } catch (err) {
          console.error('Inference error:', err)
          setError(err instanceof Error ? err.message : 'Inference failed')
        } finally {
          setIsInferring(false)
        }
      }
      reader.readAsDataURL(selectedImage)
    } catch (err) {
      console.error('Error reading image:', err)
      setError('Failed to read image file')
      setIsInferring(false)
    }
  }

  const clearImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Test Inference</h3>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Image Upload & Parameters */}
        <div className="space-y-6">
          {/* Image Upload */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Test Image
            </label>
            {!selectedImage ? (
              <div
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.click()}
                className={cn(
                  'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
                  isDragging
                    ? 'border-violet-600 bg-violet-50'
                    : 'border-gray-300 hover:border-violet-400 hover:bg-gray-50'
                )}
              >
                <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 mb-2">
                  Drag & drop an image here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports: JPG, PNG, WebP
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
              </div>
            ) : (
              <div className="relative">
                <img
                  src={imagePreview!}
                  alt="Selected"
                  className="w-full h-auto rounded-lg border border-gray-200"
                />
                <button
                  onClick={clearImage}
                  className="absolute top-2 right-2 p-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors shadow-md"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {/* Inference Parameters */}
          {taskType === 'detection' && (
            <div className="space-y-4">
              <h4 className="text-sm font-semibold text-gray-900">Inference Parameters</h4>

              {/* Confidence Threshold */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Confidence Threshold
                  </label>
                  <span className="text-sm font-mono text-gray-900">{confidenceThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Minimum confidence score for detections
                </p>
              </div>

              {/* IOU Threshold */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    IOU Threshold
                  </label>
                  <span className="text-sm font-mono text-gray-900">{iouThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <p className="text-xs text-gray-500 mt-1">
                  IOU threshold for non-maximum suppression
                </p>
              </div>

              {/* Max Detections */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">
                    Max Detections
                  </label>
                  <span className="text-sm font-mono text-gray-900">{maxDetections}</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="300"
                  step="10"
                  value={maxDetections}
                  onChange={(e) => setMaxDetections(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Maximum number of detections to return
                </p>
              </div>
            </div>
          )}

          {/* Run Inference Button */}
          <button
            onClick={handleInference}
            disabled={!selectedImage || isInferring}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
          >
            {isInferring ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Running Inference...
              </>
            ) : (
              <>
                <PlayCircle className="w-5 h-5" />
                Run Inference
              </>
            )}
          </button>
        </div>

        {/* Right Column: Results */}
        <div className="space-y-6">
          <h4 className="text-sm font-semibold text-gray-900">Results</h4>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800">Error</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            </div>
          )}

          {!result && !error && (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <ImageIcon className="w-16 h-16 text-gray-300 mb-4" />
              <p className="text-gray-600 mb-2">No results yet</p>
              <p className="text-sm text-gray-500">
                Upload an image and run inference to see results
              </p>
            </div>
          )}

          {result && (
            <>
              {/* Inference Stats */}
              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-blue-900">Inference Time</span>
                  <span className="text-lg font-semibold text-blue-900">
                    {result.inference_time_ms.toFixed(2)} ms
                  </span>
                </div>
              </div>

              {/* Canvas with drawn detections */}
              {result.detections && result.detections.length > 0 && (
                <div className="border border-gray-200 rounded-lg overflow-hidden">
                  <canvas ref={canvasRef} className="w-full h-auto" />
                </div>
              )}

              {/* Detection Results */}
              {result.detections && result.detections.length > 0 ? (
                <div>
                  <h5 className="text-sm font-semibold text-gray-900 mb-3">
                    Detections ({result.detections.length})
                  </h5>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {result.detections.map((det, idx) => (
                      <div
                        key={idx}
                        className="p-3 bg-gray-50 border border-gray-200 rounded-lg"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">
                            {det.class_name || `Class ${det.class_id}`}
                          </span>
                          <span className="text-sm font-semibold text-green-600">
                            {(det.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="grid grid-cols-4 gap-2 text-xs text-gray-600">
                          <div>
                            <span className="font-medium">x1:</span> {det.bbox.x1.toFixed(0)}
                          </div>
                          <div>
                            <span className="font-medium">y1:</span> {det.bbox.y1.toFixed(0)}
                          </div>
                          <div>
                            <span className="font-medium">x2:</span> {det.bbox.x2.toFixed(0)}
                          </div>
                          <div>
                            <span className="font-medium">y2:</span> {det.bbox.y2.toFixed(0)}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg text-center">
                  <p className="text-sm text-gray-600">No detections found</p>
                  <p className="text-xs text-gray-500 mt-1">
                    Try lowering the confidence threshold
                  </p>
                </div>
              )}

              {/* Classification Result */}
              {result.classification_result && (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h5 className="text-sm font-semibold text-green-900 mb-2">Classification Result</h5>
                  <div className="flex items-center justify-between">
                    <span className="text-lg font-semibold text-green-900">
                      {result.classification_result.class_name}
                    </span>
                    <span className="text-lg font-semibold text-green-600">
                      {(result.classification_result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
