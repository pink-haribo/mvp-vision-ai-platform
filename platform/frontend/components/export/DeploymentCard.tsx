'use client'

import { useState } from 'react'
import { Rocket, Copy, TestTube, Power, Trash2, CheckCircle, XCircle, Clock, ExternalLink, Package, Container, Download } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

export interface Deployment {
  id: number
  export_job_id: number
  deployment_type: 'platform_endpoint' | 'edge_package' | 'container' | 'download'
  status: 'active' | 'inactive' | 'failed'
  deployment_config: {
    endpoint_url?: string
    api_key?: string
    container_image?: string
    registry?: string
    package_url?: string
    download_url?: string
  }
  created_at: string
  activated_at?: string
  deactivated_at?: string
  usage_stats?: {
    request_count: number
    total_inference_time_ms: number
    avg_latency_ms: number
    last_request_at?: string
  }
  error_message?: string
}

interface DeploymentCardProps {
  deployment: Deployment
  onTest?: () => void
  onDeactivate?: () => void
  onActivate?: () => void
  onDelete?: () => void
}

const typeInfo: Record<string, { name: string; icon: React.ReactNode; color: string; description: string }> = {
  platform_endpoint: {
    name: 'Platform Endpoint',
    icon: <Rocket className="w-5 h-5" />,
    color: 'violet',
    description: 'Real-time inference API on platform infrastructure'
  },
  edge_package: {
    name: 'Edge Package',
    icon: <Package className="w-5 h-5" />,
    color: 'blue',
    description: 'Optimized package for edge device deployment'
  },
  container: {
    name: 'Container Image',
    icon: <Container className="w-5 h-5" />,
    color: 'green',
    description: 'Docker container for self-hosted deployment'
  },
  download: {
    name: 'Direct Download',
    icon: <Download className="w-5 h-5" />,
    color: 'gray',
    description: 'Download exported model files'
  }
}

const colorClasses: Record<string, { bg: string; text: string; border: string }> = {
  violet: { bg: 'bg-violet-100', text: 'text-violet-800', border: 'border-violet-200' },
  blue: { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-200' },
  green: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-200' },
  gray: { bg: 'bg-gray-100', text: 'text-gray-800', border: 'border-gray-200' }
}

export default function DeploymentCard({ deployment, onTest, onDeactivate, onActivate, onDelete }: DeploymentCardProps) {
  const [copiedField, setCopiedField] = useState<string | null>(null)

  const typeDetails = typeInfo[deployment.deployment_type]
  const colors = colorClasses[typeDetails?.color || 'gray']

  const copyToClipboard = async (text: string, field: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedField(field)
      setTimeout(() => setCopiedField(null), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const getStatusBadge = () => {
    switch (deployment.status) {
      case 'active':
        return (
          <div className="flex items-center gap-2 text-green-600">
            <CheckCircle className="w-4 h-4" />
            <span className="text-sm font-medium">Active</span>
          </div>
        )
      case 'inactive':
        return (
          <div className="flex items-center gap-2 text-gray-600">
            <Clock className="w-4 h-4" />
            <span className="text-sm font-medium">Inactive</span>
          </div>
        )
      case 'failed':
        return (
          <div className="flex items-center gap-2 text-red-600">
            <XCircle className="w-4 h-4" />
            <span className="text-sm font-medium">Failed</span>
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

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
    return num.toString()
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5 hover:shadow-md transition-shadow">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={cn('p-2 rounded-lg', colors.bg, colors.text)}>
            {typeDetails?.icon}
          </div>
          <div>
            <h4 className="font-semibold text-gray-900">{typeDetails?.name}</h4>
            <p className="text-xs text-gray-500 mt-0.5">{typeDetails?.description}</p>
          </div>
        </div>
        {getStatusBadge()}
      </div>

      {/* Platform Endpoint Details */}
      {deployment.deployment_type === 'platform_endpoint' && deployment.deployment_config.endpoint_url && (
        <div className="space-y-3 mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-medium text-gray-600">Endpoint URL</label>
              <button
                onClick={() => copyToClipboard(deployment.deployment_config.endpoint_url!, 'endpoint')}
                className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded transition-colors"
                title="Copy endpoint URL"
              >
                {copiedField === 'endpoint' ? (
                  <CheckCircle className="w-3.5 h-3.5 text-green-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            <code className="text-xs text-gray-800 bg-white px-2 py-1 rounded border border-gray-200 block overflow-x-auto">
              {deployment.deployment_config.endpoint_url}
            </code>
          </div>

          {deployment.deployment_config.api_key && (
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">API Key</label>
                <button
                  onClick={() => copyToClipboard(deployment.deployment_config.api_key!, 'apikey')}
                  className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded transition-colors"
                  title="Copy API key"
                >
                  {copiedField === 'apikey' ? (
                    <CheckCircle className="w-3.5 h-3.5 text-green-600" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                </button>
              </div>
              <code className="text-xs text-gray-800 bg-white px-2 py-1 rounded border border-gray-200 block overflow-x-auto font-mono">
                {deployment.deployment_config.api_key.substring(0, 20)}...
              </code>
            </div>
          )}
        </div>
      )}

      {/* Container Details */}
      {deployment.deployment_type === 'container' && deployment.deployment_config.container_image && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs font-medium text-gray-600">Container Image</label>
              <button
                onClick={() => copyToClipboard(deployment.deployment_config.container_image!, 'image')}
                className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded transition-colors"
                title="Copy image name"
              >
                {copiedField === 'image' ? (
                  <CheckCircle className="w-3.5 h-3.5 text-green-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            <code className="text-xs text-gray-800 bg-white px-2 py-1 rounded border border-gray-200 block overflow-x-auto">
              {deployment.deployment_config.container_image}
            </code>
          </div>
          {deployment.deployment_config.registry && (
            <p className="text-xs text-gray-500 mt-2">
              Registry: {deployment.deployment_config.registry}
            </p>
          )}
        </div>
      )}

      {/* Edge Package / Download Details */}
      {(deployment.deployment_type === 'edge_package' || deployment.deployment_type === 'download') &&
       (deployment.deployment_config.package_url || deployment.deployment_config.download_url) && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <a
            href={deployment.deployment_config.package_url || deployment.deployment_config.download_url}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            <ExternalLink className="w-4 h-4" />
            Download Package
          </a>
        </div>
      )}

      {/* Usage Stats */}
      {deployment.status === 'active' && deployment.usage_stats && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h5 className="text-xs font-semibold text-blue-900 mb-2">Usage Statistics</h5>
          <div className="grid grid-cols-3 gap-3">
            <div>
              <p className="text-xs text-blue-600">Requests</p>
              <p className="text-lg font-semibold text-blue-900">
                {formatNumber(deployment.usage_stats.request_count)}
              </p>
            </div>
            <div>
              <p className="text-xs text-blue-600">Avg Latency</p>
              <p className="text-lg font-semibold text-blue-900">
                {deployment.usage_stats.avg_latency_ms.toFixed(0)}ms
              </p>
            </div>
            <div>
              <p className="text-xs text-blue-600">Total Time</p>
              <p className="text-lg font-semibold text-blue-900">
                {(deployment.usage_stats.total_inference_time_ms / 1000).toFixed(1)}s
              </p>
            </div>
          </div>
          {deployment.usage_stats.last_request_at && (
            <p className="text-xs text-blue-600 mt-2">
              Last request: {formatDate(deployment.usage_stats.last_request_at)}
            </p>
          )}
        </div>
      )}

      {/* Error Message */}
      {deployment.status === 'failed' && deployment.error_message && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-800 font-medium mb-1">Error:</p>
          <p className="text-sm text-red-600">{deployment.error_message}</p>
        </div>
      )}

      {/* Metadata */}
      <div className="space-y-2 mb-4 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-gray-600">Deployment ID:</span>
          <span className="font-mono text-gray-900">#{deployment.id}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-600">Created:</span>
          <span className="text-gray-900">{formatDate(deployment.created_at)}</span>
        </div>
        {deployment.activated_at && (
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Activated:</span>
            <span className="text-gray-900">{formatDate(deployment.activated_at)}</span>
          </div>
        )}
        {deployment.deactivated_at && (
          <div className="flex items-center justify-between">
            <span className="text-gray-600">Deactivated:</span>
            <span className="text-gray-900">{formatDate(deployment.deactivated_at)}</span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-3 border-t border-gray-100">
        {deployment.status === 'active' && deployment.deployment_type === 'platform_endpoint' && (
          <button
            onClick={onTest}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
          >
            <TestTube className="w-4 h-4" />
            Test Inference
          </button>
        )}

        {deployment.status === 'active' && (
          <button
            onClick={onDeactivate}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors text-sm font-medium"
          >
            <Power className="w-4 h-4" />
            Deactivate
          </button>
        )}

        {deployment.status === 'inactive' && (
          <button
            onClick={onActivate}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium"
          >
            <Power className="w-4 h-4" />
            Activate
          </button>
        )}

        <div className="flex-1" />

        <button
          onClick={onDelete}
          className="p-2 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          title="Delete deployment"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  )
}
