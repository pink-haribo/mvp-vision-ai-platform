'use client'

import { useState, useRef } from 'react'
import { FolderOpen, Upload, X, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface DatasetFolderUploadProps {
  onSuccess?: () => void
}

export default function DatasetFolderUpload({ onSuccess }: DatasetFolderUploadProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [uploadMessage, setUploadMessage] = useState('')
  const [datasetName, setDatasetName] = useState('')
  const [description, setDescription] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [fileCount, setFileCount] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFiles(files)
      setFileCount(files.length)

      // Auto-generate dataset name from folder name
      const firstFile = files[0]
      const folderName = firstFile.webkitRelativePath.split('/')[0]
      if (!datasetName) {
        setDatasetName(folderName)
      }
    }
  }

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setUploadStatus('error')
      setUploadMessage('Please select a folder')
      return
    }

    if (!datasetName.trim()) {
      setUploadStatus('error')
      setUploadMessage('Please enter a dataset name')
      return
    }

    setUploading(true)
    setUploadStatus('idle')
    setUploadMessage('')

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const formData = new FormData()

      // Add metadata
      formData.append('dataset_name', datasetName.trim())
      if (description.trim()) {
        formData.append('description', description.trim())
      }
      formData.append('visibility', 'private')

      // Add all files
      Array.from(selectedFiles).forEach((file) => {
        // Use webkitRelativePath as filename to preserve folder structure
        const relativePath = file.webkitRelativePath || file.name
        formData.append('files', file, relativePath)
      })

      const response = await fetch(`${baseUrl}/datasets/upload-folder`, {
        method: 'POST',
        body: formData,
      })

      const result = await response.json()

      if (response.ok && result.status === 'success') {
        setUploadStatus('success')
        setUploadMessage(result.message || 'Folder uploaded successfully!')
        setTimeout(() => {
          setIsOpen(false)
          setDatasetName('')
          setDescription('')
          setSelectedFiles(null)
          setFileCount(0)
          if (fileInputRef.current) {
            fileInputRef.current.value = ''
          }
          onSuccess?.()
        }, 2000)
      } else {
        throw new Error(result.message || 'Upload failed')
      }
    } catch (error) {
      console.error('Upload error:', error)
      setUploadStatus('error')
      setUploadMessage(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          'px-4 py-2 rounded-lg',
          'bg-violet-600 hover:bg-violet-700',
          'text-white text-sm font-medium',
          'transition-colors flex items-center gap-2'
        )}
      >
        <FolderOpen className="w-4 h-4" />
        Upload Folder
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-6 h-6 text-violet-600" />
                <h2 className="text-xl font-semibold text-gray-900">Upload Dataset Folder</h2>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                disabled={uploading}
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
                  disabled={uploading}
                  className={cn(
                    'w-full px-3 py-2 border rounded-lg',
                    'focus:ring-2 focus:ring-violet-500 focus:border-transparent',
                    'disabled:bg-gray-100 disabled:cursor-not-allowed'
                  )}
                  placeholder="Enter dataset name"
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
                  disabled={uploading}
                  rows={3}
                  className={cn(
                    'w-full px-3 py-2 border rounded-lg',
                    'focus:ring-2 focus:ring-violet-500 focus:border-transparent',
                    'disabled:bg-gray-100 disabled:cursor-not-allowed'
                  )}
                  placeholder="Enter description"
                />
              </div>

              {/* Folder Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Folder
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFolderSelect}
                  disabled={uploading}
                  webkitdirectory=""
                  directory=""
                  multiple
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className={cn(
                    'w-full px-4 py-3 border-2 border-dashed rounded-lg',
                    'hover:border-violet-500 hover:bg-violet-50',
                    'transition-colors flex items-center justify-center gap-2',
                    'disabled:opacity-50 disabled:cursor-not-allowed'
                  )}
                >
                  <FolderOpen className="w-5 h-5 text-violet-600" />
                  <span className="text-sm font-medium text-gray-700">
                    {selectedFiles ? `Selected: ${fileCount} files` : 'Click to select folder'}
                  </span>
                </button>
              </div>

              {/* Upload Status */}
              {uploadMessage && (
                <div className={cn(
                  'p-4 rounded-lg flex items-start gap-3',
                  uploadStatus === 'success' && 'bg-green-50 border border-green-200',
                  uploadStatus === 'error' && 'bg-red-50 border border-red-200'
                )}>
                  {uploadStatus === 'success' && <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />}
                  {uploadStatus === 'error' && <XCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />}
                  <p className={cn(
                    'text-sm',
                    uploadStatus === 'success' && 'text-green-700',
                    uploadStatus === 'error' && 'text-red-700'
                  )}>
                    {uploadMessage}
                  </p>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="p-6 border-t flex items-center justify-end gap-3">
              <button
                onClick={() => setIsOpen(false)}
                disabled={uploading}
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
                onClick={handleUpload}
                disabled={uploading || !selectedFiles || !datasetName.trim()}
                className={cn(
                  'px-4 py-2 rounded-lg',
                  'bg-violet-600 hover:bg-violet-700',
                  'text-white text-sm font-medium',
                  'transition-colors flex items-center gap-2',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Uploading...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    Upload
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
