'use client'

/**
 * ImageUploadList Component
 *
 * Unified component for image upload and list display.
 * - Before upload: Shows drop zone
 * - After upload: Shows grid list of images
 *
 * Phase 3: Layout improvement - combines upload area and image list
 */

import { useRef } from 'react'
import { Upload, Trash2, Plus, X } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface UploadedImage {
  id: string
  file: File
  preview: string
  serverPath?: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  result?: any
  error?: string
}

interface ImageUploadListProps {
  images: UploadedImage[]
  selectedImageId: string | null
  onImagesAdd: (files: File[]) => void
  onImageSelect: (imageId: string) => void
  onImageRemove: (imageId: string) => void
  onClearAll: () => void
}

export default function ImageUploadList({
  images,
  selectedImageId,
  onImagesAdd,
  onImageSelect,
  onImageRemove,
  onClearAll,
}: ImageUploadListProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      onImagesAdd(files)
    }
    // Reset input
    e.target.value = ''
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = Array.from(e.dataTransfer.files).filter(file =>
      file.type.startsWith('image/')
    )
    if (files.length > 0) {
      onImagesAdd(files)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  // Show upload zone if no images
  if (images.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-sm font-semibold text-gray-900 mb-4">이미지 업로드</h3>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          className={cn(
            'border-2 border-dashed border-gray-300',
            'rounded-lg p-12',
            'text-center',
            'hover:border-violet-400 hover:bg-violet-50/50',
            'transition-colors cursor-pointer'
          )}
        >
          <Upload className="w-12 h-12 mx-auto mb-3 text-gray-400" />
          <p className="text-sm text-gray-600 mb-1 font-medium">
            이미지를 드래그 앤 드롭하세요
          </p>
          <p className="text-xs text-gray-500">
            또는 클릭하여 파일 선택
          </p>
          <p className="text-xs text-gray-400 mt-2">
            JPG, PNG, WEBP 지원
          </p>
        </div>
      </div>
    )
  }

  // Show grid list if images exist
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-gray-900">
            업로드된 이미지
          </h3>
          <p className="text-xs text-gray-500 mt-0.5">
            {images.length}개 업로드됨
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium',
              'bg-violet-600 text-white hover:bg-violet-700',
              'flex items-center gap-1.5',
              'transition-colors'
            )}
          >
            <Plus className="w-3.5 h-3.5" />
            추가
          </button>
          <button
            onClick={onClearAll}
            className={cn(
              'px-3 py-1.5 rounded-lg text-xs font-medium',
              'bg-gray-100 text-gray-700 hover:bg-gray-200',
              'flex items-center gap-1.5',
              'transition-colors'
            )}
          >
            <Trash2 className="w-3.5 h-3.5" />
            전체 삭제
          </button>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Grid List - 2 columns */}
      <div className="grid grid-cols-2 gap-3 max-h-[500px] overflow-y-auto">
        {images.map((image) => (
          <div
            key={image.id}
            onClick={() => {
              console.log('[ImageUploadList] Image clicked, id:', image.id)
              console.log('[ImageUploadList] Calling onImageSelect...')
              onImageSelect(image.id)
            }}
            className={cn(
              'relative group cursor-pointer rounded-lg border-2 p-2 transition-all',
              selectedImageId === image.id
                ? 'border-violet-600 bg-violet-50 shadow-md'
                : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
            )}
          >
            {/* Image */}
            <div className="relative">
              <img
                src={image.preview}
                alt={image.file.name}
                className="w-full h-24 object-cover rounded mb-2"
              />

              {/* Status Badge */}
              <div className="absolute top-1 right-1">
                <span className={cn(
                  'text-xs px-1.5 py-0.5 rounded font-medium',
                  image.status === 'completed' && 'bg-green-500 text-white',
                  image.status === 'pending' && 'bg-gray-500 text-white',
                  image.status === 'processing' && 'bg-blue-500 text-white',
                  image.status === 'failed' && 'bg-red-500 text-white'
                )}>
                  {image.status === 'completed' && '✓'}
                  {image.status === 'pending' && '⏳'}
                  {image.status === 'processing' && '⚙️'}
                  {image.status === 'failed' && '✗'}
                </span>
              </div>

              {/* Remove Button */}
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onImageRemove(image.id)
                }}
                className={cn(
                  'absolute top-1 left-1',
                  'p-1 rounded-full',
                  'bg-red-500 text-white',
                  'opacity-0 group-hover:opacity-100',
                  'transition-opacity',
                  'hover:bg-red-600'
                )}
              >
                <X className="w-3 h-3" />
              </button>
            </div>

            {/* Filename */}
            <p className="text-xs text-gray-600 truncate" title={image.file.name}>
              {image.file.name}
            </p>

            {/* Error Message */}
            {image.error && image.status === 'failed' && (
              <p className="text-xs text-red-600 mt-1 truncate" title={image.error}>
                {image.error}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Drop zone for additional uploads */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className={cn(
          'mt-3 p-4 border-2 border-dashed border-gray-200 rounded-lg',
          'text-center text-xs text-gray-500',
          'hover:border-violet-300 hover:bg-violet-50/30',
          'transition-colors cursor-pointer'
        )}
        onClick={() => fileInputRef.current?.click()}
      >
        <Upload className="w-4 h-4 mx-auto mb-1 text-gray-400" />
        여기에 이미지를 드롭하거나 클릭하여 추가
      </div>
    </div>
  )
}
