import { useState, useRef } from 'react'

interface DatasetImageUploadProps {
  datasetId: string
  onUploadSuccess?: () => void
}

export default function DatasetImageUpload({ datasetId, onUploadSuccess }: DatasetImageUploadProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadMessage(null)

    try {
      const file = files[0]

      // Validate file type
      if (!file.type.startsWith('image/')) {
        setUploadMessage({ type: 'error', text: 'Please select an image file' })
        return
      }

      // Create form data
      const formData = new FormData()
      formData.append('file', file)

      // Upload to backend
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
      const response = await fetch(`${baseUrl}/datasets/${datasetId}/images`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (response.ok && data.status === 'success') {
        setUploadMessage({ type: 'success', text: `Successfully uploaded ${file.name}` })
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
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Image</h3>

      <div className="space-y-4">
        {/* File Input */}
        <div>
          <label htmlFor="image-upload" className="block text-sm font-medium text-gray-700 mb-2">
            Select Image
          </label>
          <input
            ref={fileInputRef}
            id="image-upload"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            disabled={uploading}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100
              disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <p className="mt-1 text-xs text-gray-500">
            Supported formats: JPG, PNG, GIF, TIFF (max 10MB)
          </p>
        </div>

        {/* Upload Status */}
        {uploading && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600" />
            <span>Uploading...</span>
          </div>
        )}

        {/* Upload Message */}
        {uploadMessage && (
          <div
            className={`p-3 rounded-lg text-sm ${
              uploadMessage.type === 'success'
                ? 'bg-green-50 text-green-800 border border-green-200'
                : 'bg-red-50 text-red-800 border border-red-200'
            }`}
          >
            {uploadMessage.text}
          </div>
        )}
      </div>
    </div>
  )
}
