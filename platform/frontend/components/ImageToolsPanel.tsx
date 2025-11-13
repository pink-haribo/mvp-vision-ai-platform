'use client'

import { useState } from 'react'
import { Upload, Wand2, Image as ImageIcon, Sparkles, ArrowRight, X } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

type ToolType = 'super_resolution' | 'background_removal' | 'enhancement'

interface ToolOption {
  id: ToolType
  name: string
  description: string
  icon: React.ReactNode
  available: boolean
}

export default function ImageToolsPanel() {
  const [selectedTool, setSelectedTool] = useState<ToolType>('super_resolution')
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [scale, setScale] = useState<2 | 4>(2)

  const tools: ToolOption[] = [
    {
      id: 'super_resolution',
      name: '초해상화 (Super-Resolution)',
      description: '이미지 해상도를 2배 또는 4배로 향상',
      icon: <Sparkles className="w-5 h-5" />,
      available: true,
    },
    {
      id: 'background_removal',
      name: '배경 제거',
      description: '자동으로 배경 제거 및 투명화',
      icon: <ImageIcon className="w-5 h-5" />,
      available: false, // Coming soon
    },
    {
      id: 'enhancement',
      name: '이미지 향상',
      description: '노이즈 제거, 선명도 향상',
      icon: <Wand2 className="w-5 h-5" />,
      available: false, // Coming soon
    },
  ]

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드 가능합니다')
      return
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('파일 크기는 10MB 이하여야 합니다')
      return
    }

    setSelectedImage(file)
    setError(null)
    setResultUrl(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreviewUrl(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleRemoveImage = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResultUrl(null)
    setError(null)
  }

  const handleProcess = async () => {
    if (!selectedImage) return

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('image', selectedImage)

      // Determine API endpoint based on selected tool
      let apiEndpoint = ''
      let queryParams = ''

      switch (selectedTool) {
        case 'super_resolution':
          apiEndpoint = '/image-tools/super-resolution'
          queryParams = `?scale=${scale}`
          break
        case 'background_removal':
          apiEndpoint = '/image-tools/background-removal'
          break
        case 'enhancement':
          apiEndpoint = '/image-tools/enhancement'
          break
        default:
          throw new Error('지원하지 않는 도구입니다')
      }

      // Call image tools API
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}${apiEndpoint}${queryParams}`,
        {
          method: 'POST',
          body: formData,
        }
      )

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || '처리 중 오류가 발생했습니다')
      }

      const result = await response.json()

      // Set result URL
      if (result.result_url) {
        setResultUrl(`${process.env.NEXT_PUBLIC_API_URL}${result.result_url}`)
      } else {
        throw new Error('결과 이미지를 받지 못했습니다')
      }
    } catch (err) {
      console.error('Processing error:', err)
      setError(err instanceof Error ? err.message : '처리 중 오류가 발생했습니다')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-6 bg-white border-b border-gray-200">
        <div className="flex items-center gap-3 mb-2">
          <Wand2 className="w-6 h-6 text-violet-600" />
          <h2 className="text-lg font-semibold text-gray-900">이미지 도구</h2>
        </div>
        <p className="text-sm text-gray-600">
          AI 기반 이미지 처리 도구를 사용하여 이미지를 향상시키세요
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Tool Selection */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 mb-3">도구 선택</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {tools.map((tool) => (
              <button
                key={tool.id}
                onClick={() => tool.available && setSelectedTool(tool.id)}
                disabled={!tool.available}
                className={cn(
                  'p-4 rounded-lg border-2 text-left transition-all',
                  'flex flex-col gap-2',
                  selectedTool === tool.id && tool.available
                    ? 'border-violet-500 bg-violet-50'
                    : 'border-gray-200 bg-white hover:border-violet-300',
                  !tool.available && 'opacity-50 cursor-not-allowed'
                )}
              >
                <div className="flex items-center gap-2">
                  <div className={cn(
                    'p-2 rounded-lg',
                    selectedTool === tool.id && tool.available
                      ? 'bg-violet-100 text-violet-600'
                      : 'bg-gray-100 text-gray-600'
                  )}>
                    {tool.icon}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-gray-900">{tool.name}</p>
                    {!tool.available && (
                      <span className="text-xs text-amber-600 font-medium">Coming Soon</span>
                    )}
                  </div>
                </div>
                <p className="text-xs text-gray-600">{tool.description}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Super-resolution Scale Selection */}
        {selectedTool === 'super_resolution' && (
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">업스케일 배율</h3>
            <div className="flex gap-3">
              <button
                onClick={() => setScale(2)}
                className={cn(
                  'flex-1 px-4 py-3 rounded-lg border-2 font-medium transition-all',
                  scale === 2
                    ? 'border-violet-500 bg-violet-50 text-violet-700'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-violet-300'
                )}
              >
                2x 업스케일
                <span className="block text-xs mt-1 font-normal opacity-75">
                  (빠름, 균형잡힌 품질)
                </span>
              </button>
              <button
                onClick={() => setScale(4)}
                className={cn(
                  'flex-1 px-4 py-3 rounded-lg border-2 font-medium transition-all',
                  scale === 4
                    ? 'border-violet-500 bg-violet-50 text-violet-700'
                    : 'border-gray-200 bg-white text-gray-700 hover:border-violet-300'
                )}
              >
                4x 업스케일
                <span className="block text-xs mt-1 font-normal opacity-75">
                  (느림, 최고 품질)
                </span>
              </button>
            </div>
          </div>
        )}

        {/* Image Upload */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 mb-3">이미지 업로드</h3>

          {!selectedImage ? (
            <label className={cn(
              'block p-8 border-2 border-dashed rounded-lg',
              'hover:border-violet-400 hover:bg-violet-50',
              'transition-colors cursor-pointer',
              'border-gray-300 bg-white'
            )}>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              <div className="flex flex-col items-center gap-3 text-center">
                <Upload className="w-10 h-10 text-gray-400" />
                <div>
                  <p className="text-sm font-medium text-gray-900">
                    클릭하여 이미지 업로드
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    JPG, PNG, WEBP 지원 (최대 10MB)
                  </p>
                </div>
              </div>
            </label>
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex items-start gap-4">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-32 h-32 object-cover rounded-lg border border-gray-200"
                  />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{selectedImage.name}</p>
                  <p className="text-xs text-gray-500 mt-1">
                    {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                  <button
                    onClick={handleProcess}
                    disabled={isProcessing}
                    className={cn(
                      'mt-3 px-4 py-2 rounded-lg font-semibold text-sm',
                      'bg-violet-600 text-white hover:bg-violet-700',
                      'disabled:opacity-50 disabled:cursor-not-allowed',
                      'transition-colors flex items-center gap-2'
                    )}
                  >
                    {isProcessing ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        처리 중...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        처리 시작
                      </>
                    )}
                  </button>
                </div>
                <button
                  onClick={handleRemoveImage}
                  className="p-2 rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}
        </div>

        {/* Results */}
        {resultUrl && (
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">처리 결과</h3>

            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="grid grid-cols-2 gap-6">
                {/* Before */}
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2">원본 이미지</p>
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Original"
                      className="w-full h-auto rounded-lg border border-gray-200"
                    />
                  )}
                </div>

                {/* After */}
                <div>
                  <p className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                    처리된 이미지
                    <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs font-medium rounded">
                      향상됨
                    </span>
                  </p>
                  <img
                    src={resultUrl}
                    alt="Processed"
                    className="w-full h-auto rounded-lg border border-gray-200"
                  />
                </div>
              </div>

              {/* Download Button */}
              <div className="mt-4 flex justify-center">
                <a
                  href={resultUrl}
                  download="processed_image.png"
                  className={cn(
                    'px-6 py-2.5 rounded-lg font-semibold text-sm',
                    'bg-green-600 text-white hover:bg-green-700',
                    'transition-colors inline-flex items-center gap-2'
                  )}
                >
                  <ArrowRight className="w-4 h-4" />
                  다운로드
                </a>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
