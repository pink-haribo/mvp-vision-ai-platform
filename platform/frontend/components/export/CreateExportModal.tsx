'use client'

import { useState, useEffect } from 'react'
import { X, ChevronRight, ChevronLeft, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface CreateExportModalProps {
  isOpen: boolean
  onClose: () => void
  trainingJobId: number
  onSuccess: () => void
}

interface ExportCapability {
  format: string
  supported: boolean
  framework_compatible: boolean
  recommended: boolean
  requires_gpu?: boolean
  notes?: string
}

interface ExportFormData {
  format: string
  // ONNX options
  opset_version?: number
  dynamic_axes?: boolean
  // TensorRT options
  fp16?: boolean
  int8?: boolean
  max_batch_size?: number
  // CoreML options
  minimum_deployment_target?: string
  // Common options
  include_validation?: boolean
  embed_preprocessing?: boolean
}

const formatInfo: Record<string, { name: string; description: string; color: string; icon: string }> = {
  onnx: {
    name: 'ONNX',
    description: 'Cross-platform, widely supported, good for CPU inference',
    color: 'blue',
    icon: '🔵'
  },
  tensorrt: {
    name: 'TensorRT',
    description: 'NVIDIA GPU optimized, fastest inference on CUDA devices',
    color: 'green',
    icon: '🟢'
  },
  coreml: {
    name: 'CoreML',
    description: 'Apple ecosystem (iOS, macOS), on-device inference',
    color: 'purple',
    icon: '🟣'
  },
  tflite: {
    name: 'TFLite',
    description: 'Mobile & embedded devices, lightweight runtime',
    color: 'orange',
    icon: '🟠'
  },
  torchscript: {
    name: 'TorchScript',
    description: 'PyTorch native format, best for PyTorch deployment',
    color: 'red',
    icon: '🔴'
  },
  openvino: {
    name: 'OpenVINO',
    description: 'Intel hardware optimized, edge deployment',
    color: 'indigo',
    icon: '🔷'
  }
}

const colorClasses: Record<string, { bg: string; border: string; text: string; hover: string }> = {
  blue: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-800', hover: 'hover:border-blue-400' },
  green: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', hover: 'hover:border-green-400' },
  purple: { bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-800', hover: 'hover:border-purple-400' },
  orange: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-800', hover: 'hover:border-orange-400' },
  red: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800', hover: 'hover:border-red-400' },
  indigo: { bg: 'bg-indigo-50', border: 'border-indigo-200', text: 'text-indigo-800', hover: 'hover:border-indigo-400' }
}

export default function CreateExportModal({ isOpen, onClose, trainingJobId, onSuccess }: CreateExportModalProps) {
  const [step, setStep] = useState(1)
  const [capabilities, setCapabilities] = useState<ExportCapability[]>([])
  const [loadingCapabilities, setLoadingCapabilities] = useState(false)
  const [formData, setFormData] = useState<ExportFormData>({
    format: '',
    opset_version: 17,
    dynamic_axes: false,
    fp16: false,
    int8: false,
    max_batch_size: 1,
    minimum_deployment_target: 'iOS15',
    include_validation: true,
    embed_preprocessing: false
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      fetchCapabilities()
    } else {
      // Reset on close
      setStep(1)
      setFormData({
        format: '',
        opset_version: 17,
        dynamic_axes: false,
        fp16: false,
        int8: false,
        max_batch_size: 1,
        minimum_deployment_target: 'iOS15',
        include_validation: true,
        embed_preprocessing: false
      })
      setError(null)
    }
  }, [isOpen, trainingJobId])

  const fetchCapabilities = async () => {
    try {
      setLoadingCapabilities(true)
      const token = localStorage.getItem('access_token')
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/capabilities?training_job_id=${trainingJobId}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch export capabilities')
      }

      const data = await response.json()
      setCapabilities(data.supported_formats || [])
    } catch (err) {
      console.error('Error fetching capabilities:', err)
      setError('Failed to load export capabilities. Please try again.')
    } finally {
      setLoadingCapabilities(false)
    }
  }

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true)
      setError(null)

      const token = localStorage.getItem('access_token')

      // Build export config based on format
      const config: Record<string, any> = {}

      if (formData.format === 'onnx') {
        config.opset_version = formData.opset_version
        // dynamic_axes should be a dict like {"input": [0, 2, 3]} or null
        // For now, we send null when disabled, let backend use defaults when enabled
        if (formData.dynamic_axes) {
          config.dynamic_axes = {"images": [0]}  // Default: batch dimension is dynamic
        }
      } else if (formData.format === 'tensorrt') {
        config.fp16 = formData.fp16
        config.int8 = formData.int8
        config.max_batch_size = formData.max_batch_size
      } else if (formData.format === 'coreml') {
        config.minimum_deployment_target = formData.minimum_deployment_target
      }

      // Add common options
      if (formData.embed_preprocessing) {
        config.embed_preprocessing = true
      }

      const requestBody: Record<string, any> = {
        training_job_id: trainingJobId,
        export_format: formData.format,
        export_config: config
      }

      // Add validation config if enabled
      if (formData.include_validation) {
        requestBody.validation_config = {
          validate_outputs: true,
          tolerance: 0.001,
          num_samples: 10
        }
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/jobs`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify(requestBody)
        }
      )

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to create export job')
      }

      // Success
      onSuccess()
      onClose()
    } catch (err) {
      console.error('Error creating export job:', err)
      setError(err instanceof Error ? err.message : 'Failed to create export job')
    } finally {
      setIsSubmitting(false)
    }
  }

  const canProceedToStep2 = formData.format !== ''
  const canProceedToStep3 = true // All options are optional

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Create Export Job</h2>
            <p className="text-sm text-gray-500 mt-1">
              Step {step} of 3: {step === 1 ? 'Format Selection' : step === 2 ? 'Optimization Options' : 'Review & Submit'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Progress Bar */}
        <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center gap-2">
            {[1, 2, 3].map((s) => (
              <div key={s} className="flex items-center flex-1">
                <div
                  className={cn(
                    'flex-1 h-1.5 rounded-full transition-colors',
                    s <= step ? 'bg-violet-600' : 'bg-gray-200'
                  )}
                />
                {s < 3 && <ChevronRight className="w-4 h-4 text-gray-400 mx-1" />}
              </div>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-800">Error</p>
                <p className="text-sm text-red-600 mt-1">{error}</p>
              </div>
            </div>
          )}

          {/* Step 1: Format Selection */}
          {step === 1 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Export Format</h3>
                <p className="text-sm text-gray-600">
                  Choose the format based on your deployment target and hardware.
                </p>
              </div>

              {loadingCapabilities ? (
                <div className="py-12 text-center">
                  <Loader2 className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
                  <p className="text-gray-600">Loading available formats...</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.keys(formatInfo).map((format) => {
                    const info = formatInfo[format]
                    const capability = capabilities.find((c) => c.format === format)
                    // Only show as supported if capability exists and is supported
                    // If capabilities array is empty (API failed), show all as unsupported
                    const isSupported = capabilities.length > 0 && capability?.supported !== false
                    const isRecommended = capability?.recommended || false
                    const colors = colorClasses[info.color]

                    return (
                      <button
                        key={format}
                        onClick={() => isSupported && setFormData({ ...formData, format })}
                        disabled={!isSupported}
                        className={cn(
                          'relative p-4 border-2 rounded-lg text-left transition-all',
                          formData.format === format
                            ? `${colors.border} ${colors.bg} shadow-md`
                            : isSupported
                            ? `border-gray-200 hover:${colors.border} ${colors.hover}`
                            : 'border-gray-200 bg-gray-50 opacity-50 cursor-not-allowed'
                        )}
                      >
                        {isRecommended && (
                          <div className="absolute -top-2 -right-2 bg-violet-600 text-white text-xs font-medium px-2 py-1 rounded-full shadow-sm">
                            Recommended
                          </div>
                        )}
                        <div className="flex items-start gap-3">
                          <span className="text-2xl">{info.icon}</span>
                          <div className="flex-1">
                            <h4 className="font-semibold text-gray-900 mb-1">{info.name}</h4>
                            <p className="text-sm text-gray-600 mb-2">{info.description}</p>
                            {!isSupported && (
                              <p className="text-xs text-red-600 font-medium">
                                Not compatible with this model
                              </p>
                            )}
                            {capability?.requires_gpu && (
                              <p className="text-xs text-orange-600 font-medium">
                                Requires GPU
                              </p>
                            )}
                            {capability?.notes && (
                              <p className="text-xs text-gray-500 mt-1">{capability.notes}</p>
                            )}
                          </div>
                          {formData.format === format && (
                            <CheckCircle className={cn('w-5 h-5', colors.text)} />
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          )}

          {/* Step 2: Optimization Options */}
          {step === 2 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Optimization Options</h3>
                <p className="text-sm text-gray-600">
                  Configure format-specific optimizations for {formatInfo[formData.format]?.name}.
                </p>
              </div>

              {/* ONNX Options */}
              {formData.format === 'onnx' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      ONNX Opset Version
                    </label>
                    <select
                      value={formData.opset_version}
                      onChange={(e) => setFormData({ ...formData, opset_version: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    >
                      <option value={13}>13 (Compatible with older runtimes)</option>
                      <option value={14}>14</option>
                      <option value={15}>15</option>
                      <option value={16}>16</option>
                      <option value={17}>17 (Recommended)</option>
                      <option value={18}>18 (Latest)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Higher versions support more operators but may require newer runtimes.
                    </p>
                  </div>

                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="dynamic_axes"
                      checked={formData.dynamic_axes}
                      onChange={(e) => setFormData({ ...formData, dynamic_axes: e.target.checked })}
                      className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                    />
                    <div className="flex-1">
                      <label htmlFor="dynamic_axes" className="text-sm font-medium text-gray-700 cursor-pointer">
                        Enable Dynamic Axes
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Allows variable batch size and input dimensions at inference time.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* TensorRT Options */}
              {formData.format === 'tensorrt' && (
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="fp16"
                      checked={formData.fp16}
                      onChange={(e) => setFormData({ ...formData, fp16: e.target.checked })}
                      className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                    />
                    <div className="flex-1">
                      <label htmlFor="fp16" className="text-sm font-medium text-gray-700 cursor-pointer">
                        FP16 Precision (Half Precision)
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Faster inference with minimal accuracy loss. Requires GPU with FP16 support.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="int8"
                      checked={formData.int8}
                      onChange={(e) => setFormData({ ...formData, int8: e.target.checked })}
                      className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                    />
                    <div className="flex-1">
                      <label htmlFor="int8" className="text-sm font-medium text-gray-700 cursor-pointer">
                        INT8 Quantization
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Significant speedup with some accuracy loss. Requires calibration dataset.
                      </p>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Max Batch Size
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={32}
                      value={formData.max_batch_size}
                      onChange={(e) => setFormData({ ...formData, max_batch_size: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Maximum batch size for TensorRT engine optimization.
                    </p>
                  </div>
                </div>
              )}

              {/* CoreML Options */}
              {formData.format === 'coreml' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Minimum Deployment Target
                    </label>
                    <select
                      value={formData.minimum_deployment_target}
                      onChange={(e) => setFormData({ ...formData, minimum_deployment_target: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    >
                      <option value="iOS13">iOS 13 / macOS 10.15</option>
                      <option value="iOS14">iOS 14 / macOS 11</option>
                      <option value="iOS15">iOS 15 / macOS 12 (Recommended)</option>
                      <option value="iOS16">iOS 16 / macOS 13</option>
                      <option value="iOS17">iOS 17 / macOS 14 (Latest)</option>
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Newer versions support more optimizations but limit device compatibility.
                    </p>
                  </div>
                </div>
              )}

              {/* TFLite, TorchScript, OpenVINO - Generic message */}
              {['tflite', 'torchscript', 'openvino'].includes(formData.format) && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm text-blue-800">
                    {formData.format === 'tflite' && 'TFLite export will use default optimization settings for mobile deployment.'}
                    {formData.format === 'torchscript' && 'TorchScript export will preserve the model architecture as-is.'}
                    {formData.format === 'openvino' && 'OpenVINO export will optimize for Intel hardware (CPU, iGPU, VPU).'}
                  </p>
                </div>
              )}

              {/* Common Options */}
              <div className="pt-4 border-t border-gray-200 space-y-4">
                <h4 className="font-medium text-gray-900">Common Options</h4>

                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="include_validation"
                    checked={formData.include_validation}
                    onChange={(e) => setFormData({ ...formData, include_validation: e.target.checked })}
                    className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                  />
                  <div className="flex-1">
                    <label htmlFor="include_validation" className="text-sm font-medium text-gray-700 cursor-pointer">
                      Include Validation
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Run inference test to verify exported model produces correct outputs.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    id="embed_preprocessing"
                    checked={formData.embed_preprocessing}
                    onChange={(e) => setFormData({ ...formData, embed_preprocessing: e.target.checked })}
                    className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                  />
                  <div className="flex-1">
                    <label htmlFor="embed_preprocessing" className="text-sm font-medium text-gray-700 cursor-pointer">
                      Embed Preprocessing
                    </label>
                    <p className="text-xs text-gray-500 mt-1">
                      Include normalization and resizing in the exported model (if supported).
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Review & Submit */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Review Configuration</h3>
                <p className="text-sm text-gray-600">
                  Please review your export configuration before submitting.
                </p>
              </div>

              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between pb-3 border-b border-gray-200">
                  <span className="text-sm font-medium text-gray-700">Export Format</span>
                  <span className={cn(
                    'px-3 py-1 rounded-full text-sm font-medium',
                    colorClasses[formatInfo[formData.format]?.color]?.bg,
                    colorClasses[formatInfo[formData.format]?.color]?.text
                  )}>
                    {formatInfo[formData.format]?.name}
                  </span>
                </div>

                {formData.format === 'onnx' && (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Opset Version</span>
                      <span className="font-medium text-gray-900">{formData.opset_version}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Dynamic Axes</span>
                      <span className="font-medium text-gray-900">{formData.dynamic_axes ? 'Enabled' : 'Disabled'}</span>
                    </div>
                  </>
                )}

                {formData.format === 'tensorrt' && (
                  <>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">FP16 Precision</span>
                      <span className="font-medium text-gray-900">{formData.fp16 ? 'Enabled' : 'Disabled'}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">INT8 Quantization</span>
                      <span className="font-medium text-gray-900">{formData.int8 ? 'Enabled' : 'Disabled'}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Max Batch Size</span>
                      <span className="font-medium text-gray-900">{formData.max_batch_size}</span>
                    </div>
                  </>
                )}

                {formData.format === 'coreml' && (
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Minimum Deployment Target</span>
                    <span className="font-medium text-gray-900">{formData.minimum_deployment_target}</span>
                  </div>
                )}

                <div className="pt-3 border-t border-gray-200 space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Include Validation</span>
                    <span className="font-medium text-gray-900">{formData.include_validation ? 'Yes' : 'No'}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">Embed Preprocessing</span>
                    <span className="font-medium text-gray-900">{formData.embed_preprocessing ? 'Yes' : 'No'}</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <p className="text-sm text-blue-800">
                  The export job will be queued and processed in the background. You can monitor progress in the Export Jobs list.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div>
            {step > 1 && (
              <button
                onClick={() => setStep(step - 1)}
                disabled={isSubmitting}
                className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              Cancel
            </button>

            {step < 3 ? (
              <button
                onClick={() => setStep(step + 1)}
                disabled={step === 1 ? !canProceedToStep2 : !canProceedToStep3}
                className="flex items-center gap-2 px-6 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="flex items-center gap-2 px-6 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors disabled:opacity-50"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    Create Export Job
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
