import { useState, useEffect } from 'react'
import Image from 'next/image'

interface ImageInfo {
  filename: string
  presigned_url: string
  size?: number | null
}

interface DatasetImageGalleryProps {
  datasetId: string
}

export default function DatasetImageGallery({ datasetId }: DatasetImageGalleryProps) {
  const [images, setImages] = useState<ImageInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<ImageInfo | null>(null)

  useEffect(() => {
    const fetchImages = async () => {
      try {
        setLoading(true)
        setError(null)

        const baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1'
        const token = localStorage.getItem('access_token')

        if (!token) {
          setError('로그인이 필요합니다.')
          setLoading(false)
          return
        }

        const response = await fetch(`${baseUrl}/datasets/${datasetId}/images`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!response.ok) {
          if (response.status === 401) {
            throw new Error('로그인이 필요합니다.')
          }
          throw new Error(`Failed to fetch images: ${response.statusText}`)
        }

        const data = await response.json()
        setImages(data.images || [])
      } catch (err) {
        console.error('Error fetching images:', err)
        setError(err instanceof Error ? err.message : 'Failed to load images')
      } finally {
        setLoading(false)
      }
    }

    if (datasetId) {
      fetchImages()
    }
  }, [datasetId])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800 text-sm">Error: {error}</p>
      </div>
    )
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <p className="mt-2 text-sm text-gray-600">No images uploaded yet</p>
        <p className="text-xs text-gray-500 mt-1">Upload images to see them here</p>
      </div>
    )
  }

  return (
    <div>
      {/* Image Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
        {images.map((image) => (
          <div
            key={image.filename}
            className="relative aspect-square bg-gray-100 rounded-lg overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all"
            onClick={() => setSelectedImage(image)}
          >
            <Image
              src={image.presigned_url}
              alt={image.filename}
              fill
              className="object-cover"
              sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, 16vw"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 p-1">
              <p className="text-white text-xs truncate">{image.filename}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-75 z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-[90vh] w-full">
            <button
              className="absolute -top-10 right-0 text-white hover:text-gray-300"
              onClick={() => setSelectedImage(null)}
            >
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="relative bg-white rounded-lg overflow-hidden">
              <Image
                src={selectedImage.presigned_url}
                alt={selectedImage.filename}
                width={1200}
                height={800}
                className="w-full h-auto"
                style={{ maxHeight: '80vh', objectFit: 'contain' }}
              />
              <div className="p-4 bg-gray-50 border-t">
                <p className="font-medium text-gray-900">{selectedImage.filename}</p>
                {selectedImage.size && (
                  <p className="text-sm text-gray-600 mt-1">
                    Size: {(selectedImage.size / 1024).toFixed(2)} KB
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
