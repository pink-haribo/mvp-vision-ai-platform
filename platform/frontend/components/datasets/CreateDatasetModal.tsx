'use client'

import { useState } from 'react'
import { X, Loader2, CheckCircle, XCircle, FolderPlus } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import { useAuth } from '@/contexts/AuthContext'

interface CreateDatasetModalProps {
  isOpen: boolean
  onClose: () => void
  onSuccess?: (datasetId: string) => void
}

export default function CreateDatasetModal({ isOpen, onClose, onSuccess }: CreateDatasetModalProps) {
  const { accessToken } = useAuth()
  const [creating, setCreating] = useState(false)
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [message, setMessage] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [description, setDescription] = useState('')
  const [visibility, setVisibility] = useState<'private' | 'public'>('private')

  const handleCreate = async () => {
    if (!datasetName.trim()) {
      setStatus('error')
      setMessage('Please enter a dataset name')
      return
    }

    setCreating(true)
    setStatus('idle')
    setMessage('')

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

      if (!accessToken) {
        setStatus('error')
        setMessage('로그인이 필요합니다.')
        setCreating(false)
        return
      }

      const response = await fetch(`${baseUrl}/datasets`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          name: datasetName.trim(),
          description: description.trim() || undefined,
          visibility,
        }),
      })

      const result = await response.json()

      if (response.ok && result.status === 'success') {
        setStatus('success')
        setMessage(result.message || 'Dataset created successfully!')

        // Call onSuccess callback if provided
        onSuccess?.(result.dataset_id)

        // Close modal after a short delay
        setTimeout(() => {
          handleClose()
        }, 1000)
      } else {
        throw new Error(result.message || 'Failed to create dataset')
      }
    } catch (error) {
      console.error('Create error:', error)
      setStatus('error')
      setMessage(error instanceof Error ? error.message : 'Failed to create dataset')
    } finally {
      setCreating(false)
    }
  }

  const handleClose = () => {
    setDatasetName('')
    setDescription('')
    setVisibility('private')
    setStatus('idle')
    setMessage('')
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b">
          <div className="flex items-center gap-3">
            <FolderPlus className="w-6 h-6 text-indigo-600" />
            <h2 className="text-xl font-semibold text-gray-900">Create New Dataset</h2>
          </div>
          <button
            onClick={handleClose}
            disabled={creating}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          {/* Dataset Name */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset Name *
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              disabled={creating}
              className={cn(
                'w-full px-3 py-2 border rounded-lg',
                'focus:ring-2 focus:ring-indigo-500 focus:border-transparent',
                'disabled:bg-gray-100 disabled:cursor-not-allowed'
              )}
              placeholder="e.g., My Training Dataset"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              disabled={creating}
              rows={3}
              className={cn(
                'w-full px-3 py-2 border rounded-lg',
                'focus:ring-2 focus:ring-indigo-500 focus:border-transparent',
                'disabled:bg-gray-100 disabled:cursor-not-allowed'
              )}
              placeholder="Describe what this dataset is for..."
            />
          </div>

          {/* Visibility */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Visibility
            </label>
            <div className="flex gap-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="private"
                  checked={visibility === 'private'}
                  onChange={(e) => setVisibility(e.target.value as 'private')}
                  disabled={creating}
                  className="mr-2"
                />
                <span className="text-sm text-gray-700">Private</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="public"
                  checked={visibility === 'public'}
                  onChange={(e) => setVisibility(e.target.value as 'public')}
                  disabled={creating}
                  className="mr-2"
                />
                <span className="text-sm text-gray-700">Public</span>
              </label>
            </div>
          </div>

          {/* Info Box */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> This will create an empty dataset. You can upload images later from the dataset page.
            </p>
          </div>

          {/* Status Message */}
          {message && (
            <div className={cn(
              'p-4 rounded-lg flex items-start gap-3',
              status === 'success' && 'bg-green-50 border border-green-200',
              status === 'error' && 'bg-red-50 border border-red-200'
            )}>
              {status === 'success' && <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />}
              {status === 'error' && <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />}
              <p className={cn(
                'text-sm',
                status === 'success' && 'text-green-700',
                status === 'error' && 'text-red-700'
              )}>
                {message}
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t flex items-center justify-end gap-3">
          <button
            onClick={handleClose}
            disabled={creating}
            className={cn(
              'px-4 py-2 rounded-lg',
              'bg-gray-100 hover:bg-gray-200',
              'text-gray-700 text-sm font-medium',
              'transition-colors',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !datasetName.trim()}
            className={cn(
              'px-4 py-2 rounded-lg',
              'bg-indigo-600 hover:bg-indigo-700',
              'text-white text-sm font-medium',
              'transition-colors flex items-center gap-2',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            {creating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <FolderPlus className="w-4 h-4" />
                Create Dataset
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
