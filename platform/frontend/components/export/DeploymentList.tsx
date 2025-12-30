'use client'

import { useState, useEffect } from 'react'
import DeploymentCard, { Deployment } from './DeploymentCard'
import { Loader2, AlertCircle, Filter } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { useAuth } from '@/contexts/AuthContext'

interface DeploymentListProps {
  trainingJobId: number
  onCreateDeployment?: () => void
  onTestInference?: (deploymentId: number) => void
}

type DeploymentTypeFilter = 'all' | 'platform_endpoint' | 'edge_package' | 'container' | 'download'
type DeploymentStatusFilter = 'all' | 'active' | 'inactive' | 'failed'

export default function DeploymentList({ trainingJobId, onCreateDeployment, onTestInference }: DeploymentListProps) {
  const { accessToken } = useAuth()
  const [deployments, setDeployments] = useState<Deployment[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [typeFilter, setTypeFilter] = useState<DeploymentTypeFilter>('all')
  const [statusFilter, setStatusFilter] = useState<DeploymentStatusFilter>('all')

  const fetchDeployments = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/deployments?training_job_id=${trainingJobId}`,
        {
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to fetch deployments')
      }

      const data = await response.json()
      setDeployments(data.deployments || [])
    } catch (err) {
      console.error('Error fetching deployments:', err)
      setError(err instanceof Error ? err.message : 'Failed to load deployments')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDeployments()
  }, [trainingJobId])

  const handleActivate = async (deploymentId: number) => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/deployments/${deploymentId}/activate`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to activate deployment')
      }

      // Refresh list
      await fetchDeployments()
    } catch (err) {
      console.error('Error activating deployment:', err)
      alert('Failed to activate deployment. Please try again.')
    }
  }

  const handleDeactivate = async (deploymentId: number) => {
    if (!confirm('Are you sure you want to deactivate this deployment?')) {
      return
    }

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/deployments/${deploymentId}/deactivate`,
        {
          method: 'PATCH',
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to deactivate deployment')
      }

      // Refresh list
      await fetchDeployments()
    } catch (err) {
      console.error('Error deactivating deployment:', err)
      alert('Failed to deactivate deployment. Please try again.')
    }
  }

  const handleDelete = async (deploymentId: number) => {
    if (!confirm('Are you sure you want to delete this deployment? This action cannot be undone.')) {
      return
    }

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/export/deployments/${deploymentId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${accessToken}`
          }
        }
      )

      if (!response.ok) {
        throw new Error('Failed to delete deployment')
      }

      // Refresh list
      await fetchDeployments()
    } catch (err) {
      console.error('Error deleting deployment:', err)
      alert('Failed to delete deployment. Please try again.')
    }
  }

  // Filter deployments
  const filteredDeployments = deployments.filter((deployment) => {
    if (typeFilter !== 'all' && deployment.deployment_type !== typeFilter) {
      return false
    }
    if (statusFilter !== 'all' && deployment.status !== statusFilter) {
      return false
    }
    return true
  })

  if (isLoading && deployments.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <Loader2 className="w-8 h-8 text-gray-400 animate-spin mx-auto mb-4" />
        <p className="text-gray-600">Loading deployments...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg border border-red-200 p-8 text-center">
        <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-4" />
        <p className="text-red-600 mb-2">Failed to load deployments</p>
        <p className="text-sm text-gray-500">{error}</p>
        <button
          onClick={fetchDeployments}
          className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
        >
          Retry
        </button>
      </div>
    )
  }

  if (deployments.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
        <div className="text-gray-400 text-4xl mb-4">🚀</div>
        <p className="text-gray-600 mb-2">No deployments yet</p>
        <p className="text-sm text-gray-500 mb-4">
          Deploy your exported model to platform endpoints, edge devices, or containers
        </p>
        <button
          onClick={onCreateDeployment}
          className="px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-700 transition-colors text-sm font-medium"
        >
          + Create Deployment
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Filter Bar */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-gray-700">
            <Filter className="w-4 h-4" />
            <span className="text-sm font-medium">Filters:</span>
          </div>

          {/* Type Filter */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Type:</label>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as DeploymentTypeFilter)}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              <option value="all">All Types</option>
              <option value="platform_endpoint">Platform Endpoint</option>
              <option value="edge_package">Edge Package</option>
              <option value="container">Container</option>
              <option value="download">Download</option>
            </select>
          </div>

          {/* Status Filter */}
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Status:</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as DeploymentStatusFilter)}
              className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              <option value="all">All Statuses</option>
              <option value="active">Active</option>
              <option value="inactive">Inactive</option>
              <option value="failed">Failed</option>
            </select>
          </div>

          <div className="flex-1" />

          {/* Results Count */}
          <div className="text-sm text-gray-600">
            {filteredDeployments.length} {filteredDeployments.length === 1 ? 'deployment' : 'deployments'}
          </div>
        </div>
      </div>

      {/* Deployments Grid */}
      {filteredDeployments.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredDeployments.map((deployment) => (
            <DeploymentCard
              key={deployment.id}
              deployment={deployment}
              onTest={() => onTestInference?.(deployment.id)}
              onActivate={() => handleActivate(deployment.id)}
              onDeactivate={() => handleDeactivate(deployment.id)}
              onDelete={() => handleDelete(deployment.id)}
            />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-gray-600 mb-2">No deployments match your filters</p>
          <p className="text-sm text-gray-500">Try adjusting the filters above</p>
        </div>
      )}

      {/* Loading indicator for ongoing refresh */}
      {isLoading && deployments.length > 0 && (
        <div className="text-center py-2">
          <span className="text-sm text-gray-500">Refreshing...</span>
        </div>
      )}
    </div>
  )
}
