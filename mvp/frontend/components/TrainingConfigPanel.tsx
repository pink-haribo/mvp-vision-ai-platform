'use client'

import { useState, useEffect } from 'react'
import { ArrowLeftIcon, ArrowRightIcon, CheckIcon, Settings } from 'lucide-react'
import { cn } from '@/lib/utils/cn'
import AdvancedConfigPanel from './training/AdvancedConfigPanel'
import ModelSelector from './training/ModelSelector'
import CustomPromptsModal from './training/CustomPromptsModal'
import { ModelInfo } from './training/ModelCard'

// Helper: Infer task type from dataset format
const getTaskTypeFromFormat = (format: string): string => {
  const taskTypeMap: Record<string, string> = {
    'imagefolder': '이미지 분류',
    'yolo': '객체 탐지',
    'coco': '객체 탐지',
  }
  return taskTypeMap[format?.toLowerCase()] || '알 수 없음'
}


interface TrainingConfig {
  framework?: string
  model_name?: string
  task_type?: string
  dataset_path?: string
  dataset_format?: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
}

interface TrainingConfigPanelProps {
  projectId?: number | null
  initialConfig?: TrainingConfig | null
  onCancel: () => void
  onTrainingStarted: (jobId: number) => void
}

export default function TrainingConfigPanel({
  projectId,
  initialConfig,
  onCancel,
  onTrainingStarted,
}: TrainingConfigPanelProps) {
  const [step, setStep] = useState(1)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Step 1: Model & Task
  const [framework, setFramework] = useState(initialConfig?.framework || 'timm')
  const [modelName, setModelName] = useState(initialConfig?.model_name || '')
  const [taskType, setTaskType] = useState(initialConfig?.task_type || 'image_classification')
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const [customPrompts, setCustomPrompts] = useState<string[]>([])
  const [showPromptsModal, setShowPromptsModal] = useState(false)

  // Step 2: Dataset
  const [datasetPath, setDatasetPath] = useState(initialConfig?.dataset_path || '')
  const [datasetFormat, setDatasetFormat] = useState(initialConfig?.dataset_format || 'imagefolder')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [datasetInfo, setDatasetInfo] = useState<any | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)

  // R2 Sample Datasets (automatically downloaded from R2 when training starts)
  const [availableDatasets, setAvailableDatasets] = useState<any[]>([
    {
      name: 'COCO8 (Object Detection)',
      path: 'det-coco8',
      description: '8 images COCO sample for object detection',
      format: 'yolo',
      task_type: 'object_detection',
      num_items: 8
    },
    {
      name: 'COCO128 (Object Detection)',
      path: 'det-coco128',
      description: '128 images COCO sample for object detection',
      format: 'yolo',
      task_type: 'object_detection',
      num_items: 128
    },
    {
      name: 'COCO8-Seg (Instance Segmentation)',
      path: 'seg-coco8',
      description: '8 images COCO sample for instance segmentation',
      format: 'yolo',
      task_type: 'instance_segmentation',
      num_items: 8
    },
    {
      name: 'COCO128-Seg (Instance Segmentation)',
      path: 'seg-coco128',
      description: '128 images COCO sample for instance segmentation',
      format: 'yolo',
      task_type: 'instance_segmentation',
      num_items: 128
    },
    {
      name: 'ImageNet-10 (Classification)',
      path: 'cls-imagenet-10',
      description: '10 classes ImageNet sample for image classification',
      format: 'imagefolder',
      task_type: 'image_classification',
      num_items: 100
    },
    {
      name: 'Imagenette2-160 (Classification)',
      path: 'cls-imagenette2-160',
      description: 'Imagenette 160x160 for image classification',
      format: 'imagefolder',
      task_type: 'image_classification',
      num_items: 9469
    },
  ])

  const [isLoadingDatasets, setIsLoadingDatasets] = useState(false)

  // Step 3: Hyperparameters
  const [epochs, setEpochs] = useState(initialConfig?.epochs || 50)
  const [batchSize, setBatchSize] = useState(initialConfig?.batch_size || 32)
  const [learningRate, setLearningRate] = useState(initialConfig?.learning_rate || 0.001)

  // Primary Metric Selection
  const [primaryMetric, setPrimaryMetric] = useState<string>('')
  const [primaryMetricMode, setPrimaryMetricMode] = useState<'max' | 'min'>('max')

  // Advanced Configuration
  const [advancedConfig, setAdvancedConfig] = useState<any>(null)
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false)

  // All available frameworks with their supported tasks
  const allFrameworks = [
    { value: 'timm', label: 'timm (PyTorch Image Models)', supportedTasks: ['image_classification'] },
    { value: 'ultralytics', label: 'Ultralytics YOLO', supportedTasks: ['object_detection', 'instance_segmentation', 'pose_estimation', 'image_classification'] },
  ]

  // All available models with their framework and supported tasks
  const allModels = [
    // timm models
    { value: 'resnet18', label: 'ResNet-18', framework: 'timm', supportedTasks: ['image_classification'] },
    { value: 'resnet50', label: 'ResNet-50', framework: 'timm', supportedTasks: ['image_classification'] },
    { value: 'efficientnet_b0', label: 'EfficientNet-B0', framework: 'timm', supportedTasks: ['image_classification'] },
    // Ultralytics models
    {
      value: 'yolov8n',
      label: 'YOLOv8n (Nano)',
      framework: 'ultralytics',
      supportedTasks: ['object_detection', 'instance_segmentation', 'pose_estimation', 'image_classification']
    },
    {
      value: 'yolov8s',
      label: 'YOLOv8s (Small)',
      framework: 'ultralytics',
      supportedTasks: ['object_detection', 'instance_segmentation', 'pose_estimation', 'image_classification']
    },
    {
      value: 'yolov8m',
      label: 'YOLOv8m (Medium)',
      framework: 'ultralytics',
      supportedTasks: ['object_detection', 'instance_segmentation', 'pose_estimation', 'image_classification']
    },
  ]

  // Get frameworks that support the selected task type
  const getFrameworkOptions = () => {
    if (!taskType) return allFrameworks

    return allFrameworks.filter(fw =>
      fw.supportedTasks.includes(taskType)
    )
  }

  // Get models that support both the selected task type and framework
  const getModelOptions = () => {
    if (!taskType) return []

    let models = allModels.filter(model =>
      model.supportedTasks.includes(taskType)
    )

    if (framework) {
      models = models.filter(model => model.framework === framework)
    }

    return models
  }

  // All task types
  const allTaskTypes = [
    { value: 'image_classification', label: '이미지 분류 (Image Classification)' },
    { value: 'object_detection', label: '객체 탐지 (Object Detection)' },
    { value: 'pose_estimation', label: '포즈 추정 (Pose Estimation)' },
  ]

  // Dataset format options
  const datasetFormats = [
    { value: 'imagefolder', label: 'ImageFolder (PyTorch)' },
    { value: 'yolo', label: 'YOLO Format' },
    { value: 'coco', label: 'COCO Format' },
  ]

  // Primary metric options based on task type
  const getMetricOptions = () => {
    const metricsByTask: Record<string, Array<{ value: string; label: string; mode: 'max' | 'min'; description: string }>> = {
      'image_classification': [
        { value: 'accuracy', label: 'Accuracy (정확도)', mode: 'max', description: '전체 예측의 정확도' },
        { value: 'loss', label: 'Loss (손실)', mode: 'min', description: '학습 손실 값' },
        { value: 'val_accuracy', label: 'Validation Accuracy', mode: 'max', description: '검증 데이터 정확도' },
        { value: 'val_loss', label: 'Validation Loss', mode: 'min', description: '검증 손실 값' },
      ],
      'object_detection': [
        { value: 'mAP50', label: 'mAP@0.5 (평균 정밀도)', mode: 'max', description: 'IoU 0.5 기준 평균 정밀도' },
        { value: 'mAP50-95', label: 'mAP@0.5:0.95', mode: 'max', description: 'COCO 표준 mAP' },
        { value: 'precision', label: 'Precision (정밀도)', mode: 'max', description: '탐지 정밀도' },
        { value: 'recall', label: 'Recall (재현율)', mode: 'max', description: '탐지 재현율' },
        { value: 'loss', label: 'Loss (손실)', mode: 'min', description: '학습 손실 값' },
      ],
      'instance_segmentation': [
        { value: 'mAP50', label: 'mAP@0.5 (평균 정밀도)', mode: 'max', description: 'IoU 0.5 기준 평균 정밀도' },
        { value: 'mAP50-95', label: 'mAP@0.5:0.95', mode: 'max', description: 'COCO 표준 mAP' },
        { value: 'precision', label: 'Precision (정밀도)', mode: 'max', description: '분할 정밀도' },
        { value: 'recall', label: 'Recall (재현율)', mode: 'max', description: '분할 재현율' },
        { value: 'loss', label: 'Loss (손실)', mode: 'min', description: '학습 손실 값' },
      ],
      'pose_estimation': [
        { value: 'mAP50', label: 'mAP@0.5 (평균 정밀도)', mode: 'max', description: 'IoU 0.5 기준 평균 정밀도' },
        { value: 'mAP50-95', label: 'mAP@0.5:0.95', mode: 'max', description: 'COCO 표준 mAP' },
        { value: 'precision', label: 'Precision (정밀도)', mode: 'max', description: '키포인트 정밀도' },
        { value: 'recall', label: 'Recall (재현율)', mode: 'max', description: '키포인트 재현율' },
        { value: 'loss', label: 'Loss (손실)', mode: 'min', description: '학습 손실 값' },
      ],
    }

    return metricsByTask[taskType] || metricsByTask['image_classification']
  }

  // REMOVED: These useEffect hooks were resetting modelName based on hardcoded allModels array
  // which didn't include newer models like yolo11n. Since we now use ModelSelector with API data,
  // we don't need these validation hooks. The ModelSelector ensures valid model selection.

  // Update primary metric when task type changes
  useEffect(() => {
    const metricOptions = getMetricOptions()
    // Auto-select first metric as default if not set
    if (!primaryMetric && metricOptions.length > 0) {
      setPrimaryMetric(metricOptions[0].value)
      setPrimaryMetricMode(metricOptions[0].mode)
    }
    // If current metric is not available for new task, reset to first option
    else if (primaryMetric && !metricOptions.find(m => m.value === primaryMetric)) {
      setPrimaryMetric(metricOptions[0].value)
      setPrimaryMetricMode(metricOptions[0].mode)
    }
  }, [taskType])

  // Fetch available datasets on mount
  useEffect(() => {
    fetchAvailableDatasets()
  }, [])

  const fetchAvailableDatasets = async () => {
    setIsLoadingDatasets(true)
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/list`)
      if (response.ok) {
        const data = await response.json()
        setAvailableDatasets(data.datasets || [])
      }
    } catch (error) {
      console.error('Failed to fetch datasets:', error)
    } finally {
      setIsLoadingDatasets(false)
    }
  }

  // Dataset analysis function
  const handleAnalyzeDataset = async () => {
    if (!datasetPath.trim()) {
      setAnalysisError('데이터셋 경로를 입력하세요')
      return
    }

    setIsAnalyzing(true)
    setAnalysisError(null)
    setDatasetInfo(null)

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/datasets/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: datasetPath.trim(),
          format_hint: null  // Auto-detect
        }),
      })

      const data = await response.json()

      if (data.status === 'success' && data.dataset_info) {
        setDatasetInfo(data.dataset_info)
        // Auto-fill format if detected
        if (data.dataset_info.format) {
          setDatasetFormat(data.dataset_info.format)
        }
      } else {
        setAnalysisError(data.message || '데이터셋 분석에 실패했습니다')
      }
    } catch (err) {
      console.error('Dataset analysis error:', err)
      setAnalysisError('데이터셋 분석 중 오류가 발생했습니다')
    } finally {
      setIsAnalyzing(false)
    }
  }

  // Validation
  const canProceedStep1 = framework && modelName && taskType && (
    // YOLO-World requires custom prompts
    taskType !== 'zero_shot_detection' || customPrompts.length > 0
  )
  const canProceedStep2 = datasetPath.trim() !== '' && datasetInfo !== null  // Need analysis
  const canSubmit = canProceedStep1 && canProceedStep2 && epochs > 0 && batchSize > 0 && learningRate > 0

  const handleNext = () => {
    setError(null)
    if (step < 3) {
      setStep(step + 1)
    }
  }

  const handlePrev = () => {
    setError(null)
    if (step > 1) {
      setStep(step - 1)
    }
  }

  const handleModelSelect = (model: ModelInfo) => {
    console.log('[DEBUG] handleModelSelect called with model:', model)
    console.log('[DEBUG]   model.framework:', model.framework)
    console.log('[DEBUG]   model.model_name:', model.model_name)
    console.log('[DEBUG]   model.task_types:', model.task_types)

    setSelectedModel(model)
    setFramework(model.framework)
    setModelName(model.model_name)
    setTaskType(model.task_types[0])  // Use first task type

    console.log('[DEBUG] After setState calls - new values:')
    console.log('[DEBUG]   framework:', model.framework)
    console.log('[DEBUG]   modelName:', model.model_name)

    // Apply recommended settings
    setBatchSize(model.recommended_batch_size)
    setLearningRate(model.recommended_lr)

    // Show prompts modal for YOLO-World
    if (model.task_types.includes('zero_shot_detection')) {
      setShowPromptsModal(true)
    } else {
      setCustomPrompts([])
    }
  }

  const handlePromptsConfirm = (prompts: string[]) => {
    setCustomPrompts(prompts)
    setShowPromptsModal(false)
  }

  const handleSubmit = async () => {
    if (!canSubmit) return

    setIsSubmitting(true)
    setError(null)

    try {
      const config = {
        framework,
        model_name: modelName,
        task_type: taskType,
        dataset_path: datasetPath.trim(),
        dataset_format: datasetFormat,
        num_classes: datasetInfo?.structure?.num_classes || undefined,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        primary_metric: primaryMetric || undefined,
        primary_metric_mode: primaryMetricMode,
        advanced_config: advancedConfig || undefined,
        custom_prompts: customPrompts.length > 0 ? customPrompts : undefined,
      }

      // DEBUG: Log what we're sending
      console.log('[DEBUG] Training config before submit:')
      console.log('[DEBUG]   framework:', framework)
      console.log('[DEBUG]   modelName state:', modelName)
      console.log('[DEBUG]   selectedModel:', selectedModel)
      console.log('[DEBUG]   config.model_name:', config.model_name)

      const requestBody: any = { config }
      if (projectId) {
        requestBody.project_id = projectId
      }

      console.log('[DEBUG] Request body:', JSON.stringify(requestBody, null, 2))

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/training/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        // Handle FastAPI validation errors (array of error objects)
        if (Array.isArray(errorData.detail)) {
          const messages = errorData.detail.map((err: any) => err.msg || err.type).join(', ')
          throw new Error(`Validation error: ${messages}`)
        }
        throw new Error(errorData.detail || errorData.message || '학습 작업 생성에 실패했습니다')
      }

      const job = await response.json()
      console.log('Training job created:', job)

      // Notify parent
      onTrainingStarted(job.id)
    } catch (err) {
      console.error('Error creating training job:', err)
      // Better error message extraction
      let errorMessage = '학습 작업 생성 중 오류가 발생했습니다'
      if (err instanceof Error) {
        errorMessage = err.message
      } else if (typeof err === 'string') {
        errorMessage = err
      } else if (err && typeof err === 'object') {
        errorMessage = JSON.stringify(err)
      }
      setError(errorMessage)
    } finally {
      setIsSubmitting(false)
    }
  }

  const renderStepIndicator = () => (
    <div className="flex items-center justify-center mb-6">
      {[1, 2, 3].map((stepNum) => (
        <div key={stepNum} className="flex items-center">
          <div
            className={cn(
              'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
              step >= stepNum
                ? 'bg-violet-600 text-white'
                : 'bg-gray-200 text-gray-600'
            )}
          >
            {step > stepNum ? <CheckIcon className="w-5 h-5" /> : stepNum}
          </div>
          {stepNum < 3 && (
            <div
              className={cn(
                'w-16 h-1 mx-2',
                step > stepNum ? 'bg-violet-600' : 'bg-gray-200'
              )}
            />
          )}
        </div>
      ))}
    </div>
  )

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <button
              onClick={onCancel}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeftIcon className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                {initialConfig ? '설정 복사하여 새 학습' : '새 학습 시작'}
              </h2>
              <p className="text-sm text-gray-600 mt-1">
                {step === 1 && '작업 유형과 모델을 선택하세요'}
                {step === 2 && '데이터셋 경로를 지정하세요'}
                {step === 3 && '학습 하이퍼파라미터를 설정하세요'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto">
          {renderStepIndicator()}

          {/* Error Message */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-800">{error}</p>
            </div>
          )}

          {/* Step 1: Model & Task */}
          {step === 1 && (
            <div className="space-y-6">
              {initialConfig && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm text-blue-800">
                    📋 기존 설정을 복사했습니다. 원하는 부분만 수정하세요.
                  </p>
                </div>
              )}

              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-base font-semibold text-gray-900">
                    모델 선택
                  </h3>
                  {selectedModel && (
                    <span className="text-sm text-gray-600">
                      선택됨: <span className="font-semibold text-blue-600">{selectedModel.display_name}</span>
                    </span>
                  )}
                </div>

                <ModelSelector
                  onModelSelect={handleModelSelect}
                  selectedModel={selectedModel}
                />
              </div>

              {/* YOLO-World Custom Prompts */}
              {selectedModel && selectedModel.task_types.includes('zero_shot_detection') && (
                <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h4 className="text-sm font-semibold text-purple-900 mb-1">
                        텍스트 프롬프트 설정 필요
                      </h4>
                      <p className="text-xs text-purple-700">
                        YOLO-World는 탐지할 객체를 자연어로 정의해야 합니다
                      </p>
                    </div>
                  </div>

                  {customPrompts.length > 0 ? (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-purple-900">
                          설정된 클래스: {customPrompts.length}개
                        </span>
                        <button
                          onClick={() => setShowPromptsModal(true)}
                          className="text-sm text-purple-700 hover:text-purple-900 font-medium"
                        >
                          수정
                        </button>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {customPrompts.slice(0, 5).map((prompt, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-1 rounded-md text-xs bg-purple-100 text-purple-800"
                          >
                            {prompt}
                          </span>
                        ))}
                        {customPrompts.length > 5 && (
                          <span className="px-2 py-1 rounded-md text-xs bg-purple-100 text-purple-800">
                            +{customPrompts.length - 5}개 더
                          </span>
                        )}
                      </div>
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowPromptsModal(true)}
                      className="w-full px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 transition-colors text-sm font-medium"
                    >
                      프롬프트 설정하기
                    </button>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Step 2: Dataset */}
          {step === 2 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  데이터셋 경로 <span className="text-red-500">*</span>
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={datasetPath}
                    onChange={(e) => {
                      setDatasetPath(e.target.value)
                      setDatasetInfo(null)  // Clear previous analysis
                      setAnalysisError(null)
                    }}
                    placeholder="예: C:\datasets\cls\imagenet-10"
                    className={cn(
                      'flex-1 px-4 py-2.5 border border-gray-300 rounded-lg',
                      'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                      'text-sm'
                    )}
                  />
                  <button
                    onClick={handleAnalyzeDataset}
                    disabled={isAnalyzing || !datasetPath.trim()}
                    className={cn(
                      'px-4 py-2.5 bg-violet-600 text-white rounded-lg',
                      'hover:bg-violet-700 transition-colors',
                      'text-sm font-medium whitespace-nowrap',
                      'disabled:opacity-50 disabled:cursor-not-allowed'
                    )}
                  >
                    {isAnalyzing ? '분석 중...' : '분석하기'}
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  절대 경로를 입력하거나 아래 목록에서 데이터셋을 선택하세요
                </p>
              </div>

              {/* Available Datasets List */}
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700">
                  사용 가능한 데이터셋
                </label>
                {isLoadingDatasets ? (
                  <div className="p-4 bg-gray-50 rounded-lg text-center">
                    <p className="text-sm text-gray-500">데이터셋 목록을 불러오는 중...</p>
                  </div>
                ) : availableDatasets.length === 0 ? (
                  <div className="p-4 bg-gray-50 rounded-lg text-center">
                    <p className="text-sm text-gray-500">사용 가능한 데이터셋이 없습니다</p>
                    <p className="text-xs text-gray-400 mt-1">C:\datasets 폴더에 데이터셋을 추가하세요</p>
                  </div>
                ) : (
                  <div className="max-h-48 overflow-y-auto border border-gray-200 rounded-lg">
                    {availableDatasets.map((dataset) => (
                      <button
                        key={dataset.path}
                        onClick={() => {
                          setDatasetPath(dataset.path)
                          setDatasetInfo(null)
                          setAnalysisError(null)
                        }}
                        className={cn(
                          'w-full px-4 py-3 text-left border-b border-gray-100 last:border-b-0',
                          'hover:bg-gray-50 transition-colors',
                          datasetPath === dataset.path && 'bg-violet-50 hover:bg-violet-100'
                        )}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 truncate">
                              {dataset.name}
                            </p>
                            <p className="text-xs text-gray-500 truncate">{dataset.path}</p>
                          </div>
                          <div className="ml-4 flex-shrink-0 text-xs text-gray-400">
                            {dataset.num_items && (
                              <span className="mr-2">{dataset.num_items.toLocaleString()} images</span>
                            )}
                            {dataset.size_mb && (
                              <span>{dataset.size_mb.toFixed(1)} MB</span>
                            )}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Analysis Error */}
              {analysisError && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                  <p className="text-sm text-red-800 font-medium">❌ {analysisError}</p>
                </div>
              )}

              {/* Analysis Result */}
              {datasetInfo && (
                <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-lg space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-emerald-600 font-semibold">✓ 데이터셋 분석 완료</span>
                    {datasetInfo.confidence && (
                      <span className="text-xs text-emerald-600">
                        (신뢰도: {Math.round(datasetInfo.confidence * 100)}%)
                      </span>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div>
                      <span className="text-gray-600">형식:</span>
                      <span className="ml-2 font-medium text-gray-900">
                        {datasetInfo.format?.toUpperCase() || 'Unknown'}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">작업 유형:</span>
                      <span className="ml-2 font-medium text-gray-900">
                        {getTaskTypeFromFormat(datasetInfo.format)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">클래스:</span>
                      <span className="ml-2 font-medium text-gray-900">
                        {datasetInfo.structure?.num_classes || 0}개
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">샘플 수:</span>
                      <span className="ml-2 font-medium text-gray-900">
                        {datasetInfo.structure?.num_samples?.toLocaleString() || 0}장
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-600">총 용량:</span>
                      <span className="ml-2 font-medium text-gray-900">
                        {datasetInfo.statistics?.total_size_mb?.toFixed(1) || 0} MB
                      </span>
                    </div>
                  </div>

                  {datasetInfo.structure?.class_names && datasetInfo.structure.class_names.length > 0 && (
                    <div>
                      <span className="text-xs text-gray-600">클래스:</span>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {datasetInfo.structure.class_names.slice(0, 10).map((className: string) => (
                          <span
                            key={className}
                            className="inline-flex items-center px-2 py-0.5 rounded text-xs bg-emerald-100 text-emerald-700"
                          >
                            {className}
                          </span>
                        ))}
                        {datasetInfo.structure.class_names.length > 10 && (
                          <span className="text-xs text-gray-500">
                            +{datasetInfo.structure.class_names.length - 10}개 더
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Preview Images */}
                  {datasetInfo.preview_images && datasetInfo.preview_images.length > 0 && (
                    <div>
                      <span className="text-xs text-gray-600">샘플 이미지 ({datasetInfo.preview_images.length}개)</span>
                      <div className="mt-2 grid grid-cols-5 gap-2">
                        {datasetInfo.preview_images.slice(0, 5).map((img: any, idx: number) => (
                          <div key={idx} className="relative aspect-square bg-gray-100 rounded overflow-hidden border border-gray-200">
                            {img.thumbnail ? (
                              <img
                                src={img.thumbnail}
                                alt={img.class}
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <div className="absolute inset-0 flex items-center justify-center">
                                <div className="text-center p-2">
                                  <div className="text-2xl">📁</div>
                                </div>
                              </div>
                            )}
                            <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-[10px] p-1 truncate text-center">
                              {img.class}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Warnings */}
                  {datasetInfo.quality_checks?.warnings && datasetInfo.quality_checks.warnings.length > 0 && (
                    <div className="pt-2 border-t border-emerald-200">
                      <p className="text-xs font-medium text-amber-700 mb-1">⚠️ 경고</p>
                      <ul className="text-xs text-amber-700 space-y-0.5">
                        {datasetInfo.quality_checks.warnings.map((warning: string, idx: number) => (
                          <li key={idx}>• {warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Help text when no analysis yet */}
              {!datasetInfo && !analysisError && !isAnalyzing && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <p className="text-sm text-blue-800">
                    💡 데이터셋 경로를 입력하고 <strong>분석하기</strong>를 클릭하면 자동으로 형식을 감지하고 통계를 수집합니다.
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Step 3: Hyperparameters */}
          {step === 3 && (
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Epochs <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(parseInt(e.target.value) || 0)}
                  min="1"
                  max="1000"
                  className={cn(
                    'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-sm'
                  )}
                />
                <p className="text-xs text-gray-500 mt-1">
                  학습 반복 횟수 (1-1000)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Batch Size <span className="text-red-500">*</span>
                </label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(parseInt(e.target.value))}
                  className={cn(
                    'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-sm bg-white'
                  )}
                >
                  <option value="8">8</option>
                  <option value="16">16</option>
                  <option value="32">32</option>
                  <option value="64">64</option>
                  <option value="128">128</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  한 번에 처리할 데이터 개수 (GPU 메모리에 따라 조정)
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate <span className="text-red-500">*</span>
                </label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)}
                  step="0.0001"
                  min="0.0001"
                  max="1"
                  className={cn(
                    'w-full px-4 py-2.5 border border-gray-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
                    'text-sm'
                  )}
                />
                <p className="text-xs text-gray-500 mt-1">
                  학습 속도 (0.0001-1.0, 일반적으로 0.001)
                </p>
              </div>

              {/* Primary Metric Selection */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <label className="block text-sm font-semibold text-blue-900 mb-2">
                  Primary Metric (주요 평가 지표)
                </label>
                <select
                  value={primaryMetric}
                  onChange={(e) => {
                    const selected = getMetricOptions().find(m => m.value === e.target.value)
                    if (selected) {
                      setPrimaryMetric(selected.value)
                      setPrimaryMetricMode(selected.mode)
                    }
                  }}
                  className={cn(
                    'w-full px-4 py-2.5 border border-blue-300 rounded-lg',
                    'focus:outline-none focus:ring-2 focus:ring-blue-600 focus:border-transparent',
                    'text-sm bg-white'
                  )}
                >
                  {getMetricOptions().map((metric) => (
                    <option key={metric.value} value={metric.value}>
                      {metric.label} {metric.mode === 'max' ? '↑' : '↓'}
                    </option>
                  ))}
                </select>
                <div className="mt-2 text-xs text-blue-700">
                  <p className="font-medium">
                    선택된 메트릭: <span className="font-mono">{primaryMetric}</span>
                    <span className="ml-2 px-1.5 py-0.5 bg-blue-200 rounded">
                      {primaryMetricMode === 'max' ? '최대화 ↑' : '최소화 ↓'}
                    </span>
                  </p>
                  <p className="mt-1 text-blue-600">
                    {getMetricOptions().find(m => m.value === primaryMetric)?.description || ''}
                  </p>
                </div>
              </div>

              {/* Advanced Configuration Button */}
              <div className="border-t border-gray-200 pt-6">
                <button
                  type="button"
                  onClick={() => setShowAdvancedConfig(true)}
                  className={cn(
                    'w-full flex items-center justify-center gap-2 px-4 py-3',
                    'border-2 border-dashed rounded-lg transition-colors',
                    advancedConfig
                      ? 'border-violet-300 bg-violet-50 text-violet-700 hover:bg-violet-100'
                      : 'border-gray-300 text-gray-600 hover:border-gray-400 hover:bg-gray-50'
                  )}
                >
                  <Settings className="w-5 h-5" />
                  <span className="font-medium">
                    {advancedConfig ? 'Advanced 설정 수정하기' : 'Advanced 설정 (선택사항)'}
                  </span>
                  {advancedConfig && (
                    <span className="ml-auto px-2 py-1 bg-violet-200 text-violet-800 rounded text-xs font-semibold">
                      설정됨
                    </span>
                  )}
                </button>
                {advancedConfig && (
                  <p className="text-xs text-gray-500 mt-2 text-center">
                    Optimizer, Scheduler, Augmentation 등이 설정되었습니다
                  </p>
                )}
              </div>

              {/* Summary */}
              <div className="p-4 bg-gray-50 rounded-lg">
                <h3 className="text-sm font-semibold text-gray-900 mb-3">설정 요약</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">모델:</span>
                    <span className="font-medium">{modelName}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">작업:</span>
                    <span className="font-medium">{taskType}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">데이터셋:</span>
                    <span className="font-medium text-xs truncate max-w-[200px]">
                      {datasetPath}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Epochs:</span>
                    <span className="font-medium">{epochs}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Batch Size:</span>
                    <span className="font-medium">{batchSize}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Learning Rate:</span>
                    <span className="font-medium">{learningRate}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer - Navigation Buttons */}
      <div className="p-6 border-t border-gray-200">
        <div className="max-w-2xl mx-auto flex gap-3">
          {step > 1 && (
            <button
              onClick={handlePrev}
              disabled={isSubmitting}
              className={cn(
                'flex-1 px-4 py-2.5 border border-gray-300 rounded-lg',
                'text-gray-700 font-medium hover:bg-gray-50',
                'transition-colors flex items-center justify-center gap-2',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <ArrowLeftIcon className="w-4 h-4" />
              이전
            </button>
          )}

          {step < 3 ? (
            <button
              onClick={handleNext}
              disabled={
                (step === 1 && !canProceedStep1) ||
                (step === 2 && !canProceedStep2)
              }
              className={cn(
                'flex-1 px-4 py-2.5 bg-violet-600 text-white rounded-lg',
                'font-medium hover:bg-violet-700',
                'transition-colors flex items-center justify-center gap-2',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              다음
              <ArrowRightIcon className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!canSubmit || isSubmitting}
              className={cn(
                'flex-1 px-4 py-2.5 bg-violet-600 text-white rounded-lg',
                'font-medium hover:bg-violet-700',
                'transition-colors flex items-center justify-center gap-2',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              {isSubmitting ? '학습 시작 중...' : '학습 시작 🚀'}
            </button>
          )}
        </div>
      </div>

      {/* Advanced Configuration Modal */}
      {showAdvancedConfig && (
        <AdvancedConfigPanel
          framework={framework}
          taskType={taskType}
          config={advancedConfig}
          onChange={(newConfig) => {
            setAdvancedConfig(newConfig)
            setShowAdvancedConfig(false)
          }}
          onClose={() => setShowAdvancedConfig(false)}
        />
      )}

      {/* Custom Prompts Modal (YOLO-World) */}
      <CustomPromptsModal
        isOpen={showPromptsModal}
        onClose={() => setShowPromptsModal(false)}
        onConfirm={handlePromptsConfirm}
        initialPrompts={customPrompts}
        modelName={selectedModel?.display_name}
      />
    </div>
  )
}
