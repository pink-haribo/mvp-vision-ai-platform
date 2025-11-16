'use client'

import { useState, useEffect } from 'react'
import { X, ChevronRight, ChevronLeft, CheckCircle, AlertCircle, Loader2, Rocket, Package, Container, Download } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface CreateDeploymentModalProps {
  isOpen: boolean
  onClose: () => void
  trainingJobId: number
  onSuccess: () => void
}

interface ExportJob {
  id: number
  export_format: string
  status: string
  created_at: string
  version: number
  is_default: boolean
  file_size_mb?: number
}

type DeploymentType = 'platform_endpoint' | 'edge_package' | 'container' | 'download'

interface DeploymentFormData {
  export_job_id: number
  deployment_type: DeploymentType
  // Platform endpoint config
  auto_activate?: boolean
  // Edge package config
  package_name?: string
  optimization_level?: 'speed' | 'balanced' | 'size'
  // Container config
  registry?: string
  image_name?: string
  include_runtime?: boolean
  // Download config (no extra config needed)
}

const deploymentTypeInfo: Record<DeploymentType, {
  name: string
  icon: React.ReactNode
  color: string
  description: string
  details: string
}> = {
  platform_endpoint: {
    name: 'Platform Endpoint',
    icon: <Rocket className="w-6 h-6" />,
    color: 'violet',
    description: 'Real-time inference API on platform infrastructure',
    details: 'Deploy to our managed inference servers with auto-scaling, monitoring, and usage tracking. Get instant API endpoint with authentication.'
  },
  edge_package: {
    name: 'Edge Package',
    icon: <Package className="w-6 h-6" />,
    color: 'blue',
    description: 'Optimized package for edge device deployment',
    details: 'Download a portable package with runtime dependencies for deployment on edge devices (Raspberry Pi, Jetson, mobile devices).'
  },
  container: {
    name: 'Container Image',
    icon: <Container className="w-6 h-6" />,
    color: 'green',
    description: 'Docker container for self-hosted deployment',
    details: 'Build and push a Docker image with your model and inference server. Deploy anywhere that supports containers (Kubernetes, Docker, cloud platforms).'
  },
  download: {
    name: 'Direct Download',
    icon: <Download className="w-6 h-6" />,
    color: 'gray',
    description: 'Download exported model files directly',
    details: 'Get a download link for the raw exported model files. Use this for custom integration or offline deployment.'
  }
}

const colorClasses: Record<string, { bg: string; border: string; text: string; hover: string }> = {
  violet: { bg: 'bg-violet-50', border: 'border-violet-200', text: 'text-violet-800', hover: 'hover:border-violet-400' },
  blue: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-800', hover: 'hover:border-blue-400' },
  green: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', hover: 'hover:border-green-400' },
  gray: { bg: 'bg-gray-50', border: 'border-gray-200', text: 'text-gray-800', hover: 'hover:border-gray-400' }
}

export default function CreateDeploymentModal({ isOpen, onClose, trainingJobId, onSuccess }: CreateDeploymentModalProps) {
  const [step, setStep] = useState(1)
  const [exportJobs, setExportJobs] = useState<ExportJob[]>([])
  const [loadingExports, setLoadingExports] = useState(false)
  const [formData, setFormData] = useState<DeploymentFormData>({
    export_job_id: 0,
    deployment_type: 'platform_endpoint',
    auto_activate: true,
    package_name: '',
    optimization_level: 'balanced',
    registry: 'docker.io',
    image_name: '',
    include_runtime: true
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen) {
      fetchExportJobs()
    } else {
      // Reset on close
      setStep(1)
      setFormData({
        export_job_id: 0,
        deployment_type: 'platform_endpoint',
        auto_activate: true,
        package_name: '',
        optimization_level: 'balanced',
        registry: 'docker.io',
        image_name: '',
        include_runtime: true
      })
      setError(null)
    }
  }, [isOpen, trainingJobId])

  const fetchExportJobs = async () => {
    try {
      setLoadingExports(true)
      const token = localStorage.getItem('access_token')
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/training/${trainingJobId}/exports`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch export jobs')
      }

      const data = await response.json()
      // Filter only completed exports
      const completed = data.filter((job: ExportJob) => job.status === 'completed')
      setExportJobs(completed)

      // Auto-select default export if exists
      const defaultExport = completed.find((job: ExportJob) => job.is_default)
      if (defaultExport) {
        setFormData({ ...formData, export_job_id: defaultExport.id })
      }
    } catch (err) {
      console.error('Error fetching export jobs:', err)
      setError('Failed to load export jobs. Please try again.')
    } finally {
      setLoadingExports(false)
    }
  }

  const handleSubmit = async () => {
    try {
      setIsSubmitting(true)
      setError(null)

      const token = localStorage.getItem('access_token')

      // Build deployment config based on type
      const config: Record<string, any> = {}

      if (formData.deployment_type === 'platform_endpoint') {
        config.auto_activate = formData.auto_activate
      } else if (formData.deployment_type === 'edge_package') {
        config.package_name = formData.package_name || `edge-package-${Date.now()}`
        config.optimization_level = formData.optimization_level
      } else if (formData.deployment_type === 'container') {
        config.registry = formData.registry
        config.image_name = formData.image_name || `model-inference:${Date.now()}`
        config.include_runtime = formData.include_runtime
      }

      const requestBody = {
        export_job_id: formData.export_job_id,
        deployment_type: formData.deployment_type,
        deployment_config: config
      }

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/deployments`,
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
        throw new Error(errorData.detail || 'Failed to create deployment')
      }

      // Success
      onSuccess()
      onClose()
    } catch (err) {
      console.error('Error creating deployment:', err)
      setError(err instanceof Error ? err.message : 'Failed to create deployment')
    } finally {
      setIsSubmitting(false)
    }
  }

  const canProceedToStep2 = formData.export_job_id > 0
  const canProceedToStep3 = true // Deployment type is always selected

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Create Deployment</h2>
            <p className="text-sm text-gray-500 mt-1">
              Step {step} of 3: {step === 1 ? 'Select Export' : step === 2 ? 'Deployment Type' : 'Configure & Deploy'}
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

          {/* Step 1: Select Export Job */}
          {step === 1 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Export to Deploy</h3>
                <p className="text-sm text-gray-600">
                  Choose which exported model you want to deploy.
                </p>
              </div>

              {loadingExports ? (
                <div className="py-12 text-center">
                  <Loader2 className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
                  <p className="text-gray-600">Loading completed exports...</p>
                </div>
              ) : exportJobs.length === 0 ? (
                <div className="py-12 text-center">
                  <AlertCircle className="w-8 h-8 text-orange-500 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">No completed exports found</p>
                  <p className="text-sm text-gray-500">
                    You need to export your trained model first before creating a deployment.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  {exportJobs.map((job) => (
                    <button
                      key={job.id}
                      onClick={() => setFormData({ ...formData, export_job_id: job.id })}
                      className={cn(
                        'w-full p-4 border-2 rounded-lg text-left transition-all',
                        formData.export_job_id === job.id
                          ? 'border-violet-600 bg-violet-50 shadow-md'
                          : 'border-gray-200 hover:border-violet-300'
                      )}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                              {job.export_format.toUpperCase()}
                            </span>
                            <span className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs font-medium">
                              v{job.version}
                            </span>
                            {job.is_default && (
                              <span className="px-2 py-1 bg-violet-100 text-violet-600 rounded text-xs font-medium">
                                Default
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-600">
                            <span>Export ID: #{job.id}</span>
                            {job.file_size_mb && <span>{job.file_size_mb.toFixed(2)} MB</span>}
                            <span>{new Date(job.created_at).toLocaleDateString('ko-KR')}</span>
                          </div>
                        </div>
                        {formData.export_job_id === job.id && (
                          <CheckCircle className="w-5 h-5 text-violet-600" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 2: Deployment Type */}
          {step === 2 && (
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Choose Deployment Type</h3>
                <p className="text-sm text-gray-600">
                  Select how you want to deploy your model.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {(Object.keys(deploymentTypeInfo) as DeploymentType[]).map((type) => {
                  const info = deploymentTypeInfo[type]
                  const colors = colorClasses[info.color]

                  return (
                    <button
                      key={type}
                      onClick={() => setFormData({ ...formData, deployment_type: type })}
                      className={cn(
                        'p-5 border-2 rounded-lg text-left transition-all',
                        formData.deployment_type === type
                          ? `${colors.border} ${colors.bg} shadow-md`
                          : `border-gray-200 ${colors.hover}`
                      )}
                    >
                      <div className="flex items-start gap-3 mb-3">
                        <div className={cn('p-2 rounded-lg', colors.bg, colors.text)}>
                          {info.icon}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-semibold text-gray-900 mb-1">{info.name}</h4>
                          <p className="text-sm text-gray-600">{info.description}</p>
                        </div>
                        {formData.deployment_type === type && (
                          <CheckCircle className={cn('w-5 h-5', colors.text)} />
                        )}
                      </div>
                      <p className="text-xs text-gray-500 leading-relaxed">{info.details}</p>
                    </button>
                  )
                })}
              </div>
            </div>
          )}

          {/* Step 3: Configure */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Configure Deployment</h3>
                <p className="text-sm text-gray-600">
                  Set up deployment options for {deploymentTypeInfo[formData.deployment_type].name}.
                </p>
              </div>

              {/* Platform Endpoint Config */}
              {formData.deployment_type === 'platform_endpoint' && (
                <div className="space-y-4">
                  <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-blue-800 mb-3">
                      Your model will be deployed to a managed inference endpoint with auto-scaling and monitoring.
                    </p>
                    <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
                      <li>Unique API endpoint URL</li>
                      <li>Bearer token authentication</li>
                      <li>Auto-scaling based on load</li>
                      <li>Usage tracking and analytics</li>
                    </ul>
                  </div>

                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="auto_activate"
                      checked={formData.auto_activate}
                      onChange={(e) => setFormData({ ...formData, auto_activate: e.target.checked })}
                      className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                    />
                    <div className="flex-1">
                      <label htmlFor="auto_activate" className="text-sm font-medium text-gray-700 cursor-pointer">
                        Auto-activate after deployment
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Start serving inference requests immediately after deployment completes.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Edge Package Config */}
              {formData.deployment_type === 'edge_package' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Package Name (Optional)
                    </label>
                    <input
                      type="text"
                      value={formData.package_name}
                      onChange={(e) => setFormData({ ...formData, package_name: e.target.value })}
                      placeholder="my-model-edge-package"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Leave empty to auto-generate a name.
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Optimization Level
                    </label>
                    <div className="grid grid-cols-3 gap-3">
                      {(['speed', 'balanced', 'size'] as const).map((level) => (
                        <button
                          key={level}
                          onClick={() => setFormData({ ...formData, optimization_level: level })}
                          className={cn(
                            'p-3 border-2 rounded-lg text-center transition-all',
                            formData.optimization_level === level
                              ? 'border-violet-600 bg-violet-50'
                              : 'border-gray-200 hover:border-violet-300'
                          )}
                        >
                          <p className="text-sm font-medium text-gray-900 capitalize">{level}</p>
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Speed: Fastest inference, larger package • Balanced: Good tradeoff • Size: Smallest package, slower inference
                    </p>
                  </div>
                </div>
              )}

              {/* Container Config */}
              {formData.deployment_type === 'container' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Container Registry
                    </label>
                    <select
                      value={formData.registry}
                      onChange={(e) => setFormData({ ...formData, registry: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    >
                      <option value="docker.io">Docker Hub (docker.io)</option>
                      <option value="ghcr.io">GitHub Container Registry (ghcr.io)</option>
                      <option value="gcr.io">Google Container Registry (gcr.io)</option>
                      <option value="custom">Custom Registry</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Image Name (Optional)
                    </label>
                    <input
                      type="text"
                      value={formData.image_name}
                      onChange={(e) => setFormData({ ...formData, image_name: e.target.value })}
                      placeholder="myorg/model-inference:latest"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-violet-500 focus:border-transparent"
                    />
                    <p className="text-xs text-gray-500 mt-1">
                      Leave empty to auto-generate. Include tag if needed (e.g., :latest, :v1.0).
                    </p>
                  </div>

                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      id="include_runtime"
                      checked={formData.include_runtime}
                      onChange={(e) => setFormData({ ...formData, include_runtime: e.target.checked })}
                      className="mt-1 w-4 h-4 text-violet-600 border-gray-300 rounded focus:ring-violet-500"
                    />
                    <div className="flex-1">
                      <label htmlFor="include_runtime" className="text-sm font-medium text-gray-700 cursor-pointer">
                        Include inference runtime
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Bundle FastAPI server and dependencies for ready-to-run container.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Download Config */}
              {formData.deployment_type === 'download' && (
                <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
                  <p className="text-sm text-gray-800 mb-3">
                    A presigned download link will be generated for your exported model files.
                  </p>
                  <ul className="text-sm text-gray-600 space-y-1 list-disc list-inside">
                    <li>Direct access to exported model files</li>
                    <li>Link valid for 24 hours</li>
                    <li>Use for custom integration or offline deployment</li>
                  </ul>
                </div>
              )}
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
                disabled={isSubmitting || exportJobs.length === 0}
                className="flex items-center gap-2 px-6 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors disabled:opacity-50"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Rocket className="w-4 h-4" />
                    Create Deployment
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
