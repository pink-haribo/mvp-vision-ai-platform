import { useState, useRef } from 'react'
import { FolderPlus, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'

interface DatasetImageUploadProps {
  datasetId: string
  onUploadSuccess?: () => void
}

export default function DatasetImageUpload({ datasetId, onUploadSuccess }: DatasetImageUploadProps) {
  const { accessToken } = useAuth()
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [uploadProgress, setUploadProgress] = useState<string>('')
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFolderSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadMessage(null)
    setUploadProgress(`Preparing to upload ${files.length} files...`)

    try {
      // Create form data with all files
      const formData = new FormData()

      // Add all files with their relative paths
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        formData.append('files', file, file.webkitRelativePath || file.name)
      }

      setUploadProgress(`Uploading ${files.length} files...`)

      // Upload to backend
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'

      if (!accessToken) {
        setUploadMessage({ type: 'error', text: '로그인이 필요합니다.' })
        setUploading(false)
        setUploadProgress('')
        return
      }

      const response = await fetch(`${baseUrl}/datasets/${datasetId}/upload-images`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`
        },
        body: formData,
      })

      const data = await response.json()

      if (response.ok && data.status === 'success') {
        setUploadMessage({
          type: 'success',
          text: data.message || `Successfully uploaded ${files.length} images`
        })
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
        onUploadSuccess?.()
      } else {
        setUploadMessage({ type: 'error', text: data.message || 'Upload failed' })
      }
    } catch (err) {
      console.error('Upload error:', err)
      setUploadMessage({ type: 'error', text: 'Upload failed. Please try again.' })
    } finally {
      setUploading(false)
      setUploadProgress('')
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center gap-2 mb-4">
        <FolderPlus className="w-5 h-5 text-indigo-600" />
        <h3 className="text-lg font-semibold text-gray-900">Upload Images</h3>
      </div>

      <div className="space-y-4">
        {/* Folder Input */}
        <div>
          <label htmlFor="folder-upload" className="block text-sm font-medium text-gray-700 mb-2">
            Select Image Folder
          </label>
          <input
            ref={fileInputRef}
            id="folder-upload"
            type="file"
            // @ts-ignore - webkitdirectory is not in TypeScript types
            webkitdirectory="true"
            directory="true"
            multiple
            onChange={handleFolderSelect}
            disabled={uploading}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded file:border-0
              file:text-sm file:font-semibold
              file:bg-indigo-50 file:text-indigo-700
              hover:file:bg-indigo-100
              disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <p className="mt-2 text-xs text-gray-500">
            Select a folder containing images. The folder structure will be preserved.
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Supported formats: JPG, PNG, GIF, TIFF
          </p>
        </div>

        {/* Upload Progress */}
        {uploadProgress && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>{uploadProgress}</span>
          </div>
        )}

        {/* Upload Message */}
        {uploadMessage && (
          <div
            className={`p-3 rounded-lg text-sm flex items-start gap-2 ${
              uploadMessage.type === 'success'
                ? 'bg-green-50 text-green-800 border border-green-200'
                : 'bg-red-50 text-red-800 border border-red-200'
            }`}
          >
            {uploadMessage.type === 'success' ? (
              <CheckCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            ) : (
              <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            )}
            <span>{uploadMessage.text}</span>
          </div>
        )}

        {/* Info Box */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <p className="text-xs text-blue-800">
            <strong>Tip:</strong> You can upload a folder with subfolders. If you include an{' '}
            <code className="bg-blue-100 px-1 rounded">annotation.json</code> file,
            the dataset will be marked as labeled.
          </p>
        </div>
      </div>
    </div>
  )
}
