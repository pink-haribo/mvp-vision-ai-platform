'use client'

import { useState } from 'react'
import { Download, Rocket, Trash2, Clock, CheckCircle, XCircle, Loader2, ChevronDown, ChevronUp } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

export interface ExportJob {
  id: number
  training_job_id: number
  export_format: 'onnx' | 'tensorrt' | 'coreml' | 'tflite' | 'torchscript' | 'openvino'
  framework: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  export_path?: string
  file_size_mb?: number
  validation_passed?: boolean
  error_message?: string
  created_at: string
  completed_at?: string
  version: number
  is_default: boolean
}

interface ExportJobCardProps {
  job: ExportJob
  onDownload?: () => void
  onDeploy?: () => void
  onDelete?: () => void
}

const formatLabels: Record<string, string> = {
  onnx: 'ONNX',
  tensorrt: 'TensorRT',
  coreml: 'CoreML',
  tflite: 'TFLite',
  torchscript: 'TorchScript',
  openvino: 'OpenVINO'
}

const formatColors: Record<string, string> = {
  onnx: 'bg-blue-100 text-blue-800',
  tensorrt: 'bg-green-100 text-green-800',
  coreml: 'bg-purple-100 text-purple-800',
  tflite: 'bg-orange-100 text-orange-800',
  torchscript: 'bg-red-100 text-red-800',
  openvino: 'bg-indigo-100 text-indigo-800'
}

export default function ExportJobCard({ job, onDownload, onDeploy, onDelete }: ExportJobCardProps) {
  const [showError, setShowError] = useState(false)

  const getStatusBadge = () => {
    switch (job.status) {
      case 'pending':
        return (
          <div className="flex items-center gap-2 text-gray-600">
            <Clock className="w-4 h-4" />
            <span className="text-sm">Pending</span>
          </div>
        )
      case 'running':
        return (
          <div className="flex items-center gap-2 text-blue-600">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Exporting...</span>
          </div>
        )
      case 'completed':
        return (
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm">Completed</span>
          </div>
        )
      case 'failed':
        return (
          <div className="flex items-center gap-2 text-red-600">
            <XCircle className="w-4 h-4" />
            <span className="text-sm">Failed</span>
          </div>
        )
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          {/* Format Badge */}
          <span className={cn(
            'px-3 py-1 rounded-full text-sm font-medium',
            formatColors[job.export_format] || 'bg-gray-100 text-gray-800'
          )}>
            {formatLabels[job.export_format] || job.export_format}
          </span>

          {/* Version Badge */}
          <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs font-medium">
            v{job.version}
          </span>

          {/* Default Badge */}
          {job.is_default && (
            <span className="px-2 py-1 bg-violet-100 text-violet-600 rounded text-xs font-medium">
              Default
            </span>
          )}
        </div>

        {/* Status */}
        <div>
          {getStatusBadge()}
        </div>
      </div>

      {/* Info */}
      <div className="space-y-2 mb-4">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Export ID:</span>
          <span className="font-mono text-gray-900">#{job.id}</span>
        </div>

        {job.file_size_mb != null && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Size:</span>
            <span className="font-medium text-gray-900">{job.file_size_mb.toFixed(2)} MB</span>
          </div>
        )}

        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">Created:</span>
          <span className="text-gray-900">{formatDate(job.created_at)}</span>
        </div>

        {job.completed_at && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Completed:</span>
            <span className="text-gray-900">{formatDate(job.completed_at)}</span>
          </div>
        )}

        {job.validation_passed !== undefined && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Validation:</span>
            <span className={cn(
              'font-medium',
              job.validation_passed ? 'text-green-600' : 'text-red-600'
            )}>
              {job.validation_passed ? 'Passed ✓' : 'Failed ✗'}
            </span>
          </div>
        )}
      </div>

      {/* Error Message - Collapsible */}
      {job.status === 'failed' && job.error_message && (
        <div className="mb-4 bg-red-50 border border-red-200 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowError(!showError)}
            className="w-full p-3 flex items-center justify-between text-left hover:bg-red-100 transition-colors"
          >
            <span className="text-sm text-red-800 font-medium">Error Details</span>
            {showError ? (
              <ChevronUp className="w-4 h-4 text-red-600" />
            ) : (
              <ChevronDown className="w-4 h-4 text-red-600" />
            )}
          </button>
          {showError && (
            <div className="px-3 pb-3 border-t border-red-200">
              <p className="text-sm text-red-600 mt-2 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
                {job.error_message}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 pt-3 border-t border-gray-100">
        {job.status === 'completed' && (
          <>
            <button
              onClick={onDownload}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
            >
              <Download className="w-4 h-4" />
              Download
            </button>
            <button
              onClick={onDeploy}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium"
            >
              <Rocket className="w-4 h-4" />
              Deploy
            </button>
          </>
        )}

        <div className="flex-1" />

        <button
          onClick={onDelete}
          className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          title="Delete export job"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
