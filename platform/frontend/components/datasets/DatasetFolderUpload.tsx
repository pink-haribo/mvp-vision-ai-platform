'use client'

import { useState, useRef } from 'react'
import { FolderOpen, Upload, X, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface DatasetFolderUploadProps {
  datasetId: string
  onSuccess?: () => void
}

export default function DatasetFolderUpload({ datasetId, onSuccess }: DatasetFolderUploadProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [uploadMessage, setUploadMessage] = useState('')
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null)
  const [fileCount, setFileCount] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      setSelectedFiles(files)
      setFileCount(files.length)
    }
  }

  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setUploadStatus('error')
      setUploadMessage('Please select a folder')
      return
    }

    setUploading(true)
    setUploadStatus('idle')
    setUploadMessage('')

    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const token = localStorage.getItem('access_token')

      if (!token) {
        throw new Error('No authentication token found. Please login.')
      }

      const formData = new FormData()

      // Add all files with their relative paths preserved
      Array.from(selectedFiles).forEach((file) => {
        // Use webkitRelativePath as filename to preserve folder structure
        const relativePath = file.webkitRelativePath || file.name
        formData.append('files', file, relativePath)
      })

      const response = await fetch(`${baseUrl}/datasets/${datasetId}/upload-images`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
      })

      const result = await response.json()

      if (response.ok && result.status === 'success') {
        setUploadStatus('success')
        setUploadMessage(result.message || 'Folder uploaded successfully!')
        setTimeout(() => {
          setIsOpen(false)
          setSelectedFiles(null)
          setFileCount(0)
          if (fileInputRef.current) {
            fileInputRef.current.value = ''
          }
          onSuccess?.()
        }, 2000)
      } else {
        throw new Error(result.detail || result.message || 'Upload failed')
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
          'w-full px-4 py-3 rounded-lg',
          'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700',
          'text-white text-sm font-medium',
          'transition-all duration-200 shadow-md hover:shadow-lg',
          'flex items-center justify-center gap-2'
        )}
      >
        <FolderOpen className="w-5 h-5" />
        폴더 업로드
      </button>

      {isOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-6 h-6 text-violet-600" />
                <h2 className="text-xl font-semibold text-gray-900">폴더 업로드</h2>
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
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <p className="text-sm text-blue-900 font-medium mb-2">
                  📁 폴더 구조 보존 업로드
                </p>
                <ul className="text-xs text-blue-800 space-y-1 ml-4 list-disc">
                  <li>폴더 구조가 그대로 유지됩니다</li>
                  <li>annotations.json 파일이 있으면 자동으로 레이블이 인식됩니다</li>
                  <li>YOLO, COCO, ImageFolder 등 다양한 형식을 지원합니다</li>
                </ul>
              </div>

              {/* Folder Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  폴더 선택
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  onChange={handleFolderSelect}
                  disabled={uploading}
                  {...({ webkitdirectory: "", directory: "" } as any)}
                  multiple
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={uploading}
                  className={cn(
                    'w-full px-4 py-8 border-2 border-dashed rounded-lg',
                    'hover:border-violet-500 hover:bg-violet-50',
                    'transition-all duration-200 flex flex-col items-center justify-center gap-3',
                    'disabled:opacity-50 disabled:cursor-not-allowed',
                    selectedFiles && 'border-violet-500 bg-violet-50'
                  )}
                >
                  <FolderOpen className={cn(
                    'w-12 h-12',
                    selectedFiles ? 'text-violet-600' : 'text-gray-400'
                  )} />
                  <div className="text-center">
                    {selectedFiles ? (
                      <>
                        <p className="text-base font-medium text-violet-900">
                          {fileCount}개 파일 선택됨
                        </p>
                        <p className="text-sm text-violet-700 mt-1">
                          다른 폴더를 선택하려면 클릭하세요
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="text-base font-medium text-gray-700">
                          폴더를 선택하세요
                        </p>
                        <p className="text-sm text-gray-500 mt-1">
                          클릭하여 데이터셋 폴더를 선택합니다
                        </p>
                      </>
                    )}
                  </div>
                </button>
              </div>

              {/* Upload Status */}
              {uploadMessage && (
                <div className={cn(
                  'p-4 rounded-lg flex items-start gap-3 animate-in fade-in duration-200',
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
            <div className="p-6 border-t bg-gray-50 flex items-center justify-end gap-3">
              <button
                onClick={() => setIsOpen(false)}
                disabled={uploading}
                className={cn(
                  'px-4 py-2 rounded-lg',
                  'bg-white border border-gray-300 hover:bg-gray-50',
                  'text-gray-700 text-sm font-medium',
                  'transition-colors',
                  'disabled:opacity-50 disabled:cursor-not-allowed'
                )}
              >
                취소
              </button>
              <button
                onClick={handleUpload}
                disabled={uploading || !selectedFiles}
                className={cn(
                  'px-6 py-2 rounded-lg',
                  'bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700',
                  'text-white text-sm font-medium',
                  'transition-all duration-200 shadow-md hover:shadow-lg',
                  'flex items-center gap-2',
                  'disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none'
                )}
              >
                {uploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    업로드 중...
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    업로드
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
