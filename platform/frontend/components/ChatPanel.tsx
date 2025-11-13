'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Download } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

interface Capability {
  frameworks: Array<{
    name: string
    display_name: string
    description: string
    supported: boolean
  }>
  models: Array<{
    name: string
    display_name: string
    description: string
    framework: string
    task_types: string[]
    supported: boolean
  }>
  task_types: Array<{
    name: string
    display_name: string
    description: string
    frameworks: string[]
    supported: boolean
  }>
  dataset_formats: Array<{
    name: string
    display_name: string
    description: string
    task_types: string[]
    supported: boolean
  }>
  parameters: Array<{
    name: string
    display_name: string
    description: string
    type: string
    required: boolean
    min?: number
    max?: number
    options?: string[]
    default: any
  }>
}

interface ChatPanelProps {
  sessionId: number | null
  onSessionCreated: (sessionId: number) => void
  onTrainingRequested: (jobId: number) => void
  onProjectSelected: (projectId: number) => void
}

export default function ChatPanel({
  sessionId,
  onSessionCreated,
  onTrainingRequested,
  onProjectSelected,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [capabilities, setCapabilities] = useState<Capability | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Phase 1: State for action-specific data
  const [datasetAnalysis, setDatasetAnalysis] = useState<any>(null)
  const [modelRecommendations, setModelRecommendations] = useState<any[]>([])
  const [trainingStatus, setTrainingStatus] = useState<any>(null)
  const [inferenceResults, setInferenceResults] = useState<any>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Auto-focus input after sending message
  useEffect(() => {
    if (!isLoading && messages.length > 0) {
      // Wait for DOM to update, then focus
      setTimeout(() => {
        inputRef.current?.focus()
      }, 100)
    }
  }, [isLoading, messages.length])

  // Fetch capabilities on mount
  useEffect(() => {
    const fetchCapabilities = async () => {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat/capabilities`)
        if (response.ok) {
          const data = await response.json()
          setCapabilities(data)
        }
      } catch (error) {
        console.error('Error fetching capabilities:', error)
      }
    }
    fetchCapabilities()
  }, [])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    setIsLoading(true)

    try {
      // Get authentication token
      const token = localStorage.getItem('access_token')
      if (!token) {
        alert('로그인이 필요합니다.')
        setIsLoading(false)
        return
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: input.trim(),
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to send message')
      }

      const data = await response.json()

      // Update session ID if new session was created
      if (!sessionId && data.session_id) {
        onSessionCreated(data.session_id)
      }

      // Add both user and assistant messages
      setMessages((prev) => [
        ...prev,
        data.user_message,
        data.assistant_message,
      ])

      // Clear input
      setInput('')

      // Phase 1: Update action-specific state from response
      // Clear previous data and only show new data if present in response
      setDatasetAnalysis(data.dataset_analysis || null)
      setModelRecommendations(data.model_recommendations || data.model_search_results || null)
      setTrainingStatus(data.training_status || null)
      setInferenceResults(data.inference_results || null)

      // If project was selected via chat, notify parent to show project detail
      if (data.selected_project_id) {
        console.log('Project selected via chat:', data.selected_project_id)
        onProjectSelected(data.selected_project_id)
      }

      // If training config is complete, create training job
      if (data.parsed_intent?.status === 'complete') {
        console.log('Training config ready:', data.parsed_intent.config)
        console.log('Metadata:', data.parsed_intent.metadata)
        await createTrainingJob(
          data.session_id,
          data.parsed_intent.config,
          data.parsed_intent.metadata
        )
      }
    } catch (error) {
      console.error('Error sending message:', error)
      alert('메시지 전송 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const createTrainingJob = async (sessionId: number, config: any, metadata?: any) => {
    try {
      // Get authentication token
      const token = localStorage.getItem('access_token')
      if (!token) {
        alert('로그인이 필요합니다.')
        return
      }

      const requestBody: any = {
        session_id: sessionId,
        config: config,
      }

      // Add metadata if available
      if (metadata) {
        if (metadata.project_id) requestBody.project_id = metadata.project_id
        if (metadata.experiment_name) requestBody.experiment_name = metadata.experiment_name
        if (metadata.tags) requestBody.tags = metadata.tags
        if (metadata.notes) requestBody.notes = metadata.notes
      }

      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/training/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(requestBody),
      })

      if (!response.ok) {
        throw new Error('Failed to create training job')
      }

      const job = await response.json()
      console.log('Training job created:', job)

      // Notify parent component
      onTrainingRequested(job.id)
    } catch (error) {
      console.error('Error creating training job:', error)
      alert('학습 작업 생성 실패')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleExportChat = () => {
    if (!sessionId || messages.length === 0) return

    try {
      // Build chat log text
      const logLines = []
      logLines.push('='.repeat(80))
      logLines.push(`Chat Session Log - Session ID: ${sessionId}`)
      logLines.push(`Exported: ${new Date().toLocaleString('ko-KR')}`)
      logLines.push(`Total Messages: ${messages.length}`)
      logLines.push('='.repeat(80))
      logLines.push('')

      messages.forEach((msg) => {
        const timestamp = new Date(msg.created_at).toLocaleString('ko-KR')
        const roleLabel = msg.role === 'user' ? 'USER' : 'ASSISTANT'

        logLines.push(`[${timestamp}] ${roleLabel}:`)
        logLines.push(msg.content)
        logLines.push('')
        logLines.push('-'.repeat(80))
        logLines.push('')
      })

      const logText = logLines.join('\n')

      // Create and download file
      const blob = new Blob([logText], { type: 'text/plain;charset=utf-8' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `chat_session_${sessionId}_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error('Failed to export chat:', error)
      alert('채팅 로그 다운로드에 실패했습니다.')
    }
  }

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">채팅</h2>
            <p className="text-sm text-gray-600 mt-1">
              자연어로 학습 설정을 입력하세요
            </p>
          </div>
          {sessionId && messages.length > 0 && (
            <button
              onClick={handleExportChat}
              className="px-3 py-2 text-sm text-gray-700 hover:text-violet-600 hover:bg-violet-50 rounded-lg transition-colors flex items-center gap-2"
              title="채팅 로그 다운로드"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">다운로드</span>
            </button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.length === 0 && (
          <div className="space-y-6">
            <div className="text-center text-gray-500 mt-8">
              <p className="text-sm">대화를 시작하세요</p>
              <p className="text-xs mt-2">
                예: "ResNet50으로 10개 클래스 분류 모델을 학습하고 싶어요"
              </p>
            </div>

            {/* Platform Capabilities */}
            {capabilities && (
              <div className="max-w-2xl mx-auto space-y-4">
                {/* Frameworks */}
                <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-lg p-4 border border-indigo-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">
                    지원하는 프레임워크
                  </h3>
                  <div className="space-y-2">
                    {capabilities.frameworks.map((framework) => (
                      <div
                        key={framework.name}
                        className={cn(
                          'text-xs p-2 rounded',
                          framework.supported
                            ? 'bg-white text-gray-900'
                            : 'bg-gray-100 text-gray-500'
                        )}
                      >
                        <div className="font-medium">
                          {framework.display_name}
                          {framework.supported && (
                            <span className="ml-2 text-emerald-600">✓ 사용 가능</span>
                          )}
                        </div>
                        <div className="text-gray-600 mt-0.5">{framework.description}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Task Types */}
                <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">
                    지원하는 작업 유형
                  </h3>
                  <div className="space-y-2">
                    {capabilities.task_types.map((task) => (
                      <div
                        key={task.name}
                        className={cn(
                          'text-xs p-2 rounded',
                          task.supported
                            ? 'bg-white text-gray-900'
                            : 'bg-gray-100 text-gray-500'
                        )}
                      >
                        <div className="font-medium">
                          {task.display_name}
                          {task.supported && (
                            <span className="ml-2 text-emerald-600">✓ 사용 가능</span>
                          )}
                        </div>
                        <div className="text-gray-600 mt-0.5">{task.description}</div>
                        {task.frameworks && (
                          <div className="text-gray-500 mt-0.5">
                            프레임워크: {task.frameworks.join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Supported Models */}
                <div className="bg-gradient-to-br from-violet-50 to-purple-50 rounded-lg p-4 border border-violet-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">
                    지원 가능한 모델
                  </h3>
                  <div className="space-y-2">
                    {capabilities.models.map((model) => (
                      <div
                        key={model.name}
                        className={cn(
                          'text-xs p-2 rounded',
                          model.supported
                            ? 'bg-white text-gray-900'
                            : 'bg-gray-100 text-gray-500'
                        )}
                      >
                        <div className="font-medium">
                          {model.display_name}
                          {model.supported && (
                            <span className="ml-2 text-emerald-600">✓ 사용 가능</span>
                          )}
                          <span className="ml-2 text-gray-500">
                            ({model.framework})
                          </span>
                        </div>
                        <div className="text-gray-600 mt-0.5">{model.description}</div>
                        {model.task_types && (
                          <div className="text-gray-500 mt-0.5">
                            작업: {model.task_types.join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Dataset Formats */}
                <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">
                    데이터셋 형식
                  </h3>
                  <div className="space-y-2">
                    {capabilities.dataset_formats.map((format) => (
                      <div
                        key={format.name}
                        className={cn(
                          'text-xs p-2 rounded',
                          format.supported
                            ? 'bg-white text-gray-900'
                            : 'bg-gray-100 text-gray-500'
                        )}
                      >
                        <div className="font-medium">
                          {format.display_name}
                          {format.supported && (
                            <span className="ml-2 text-emerald-600">✓ 사용 가능</span>
                          )}
                        </div>
                        <div className="text-gray-600 mt-0.5">{format.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              'flex',
              message.role === 'user' ? 'justify-end' : 'justify-start'
            )}
          >
            <div
              className={cn(
                'max-w-[80%] rounded-lg px-4 py-2.5',
                message.role === 'user'
                  ? 'bg-violet-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              )}
            >
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
            </div>
          </div>
        ))}

        {/* Phase 1: Dataset Analysis Card */}
        {datasetAnalysis && (
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4 border border-blue-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">📊 데이터셋 분석 결과</h3>
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">경로</div>
                  <div className="font-medium text-gray-900 break-all">{datasetAnalysis.path}</div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">형식</div>
                  <div className="font-medium text-gray-900">{datasetAnalysis.format}</div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">총 이미지</div>
                  <div className="font-medium text-gray-900">{datasetAnalysis.total_images?.toLocaleString()}개</div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">클래스 수</div>
                  <div className="font-medium text-gray-900">{datasetAnalysis.num_classes}개</div>
                </div>
              </div>
              {datasetAnalysis.classes && datasetAnalysis.classes.length > 0 && (
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600 mb-1">클래스 목록</div>
                  <div className="flex flex-wrap gap-1">
                    {datasetAnalysis.classes.map((cls: string, idx: number) => (
                      <span key={idx} className="px-2 py-0.5 bg-blue-100 text-blue-800 rounded text-xs">
                        {cls}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Phase 1: Model Recommendations Card - DISABLED per user request */}
        {/* Removed to prevent screen clutter when asking about supported models */}

        {/* Phase 1: Training Status Card */}
        {trainingStatus && (
          <div className="bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg p-4 border border-emerald-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">🔥 학습 상태</h3>
            <div className="space-y-2 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">Job ID</div>
                  <div className="font-medium text-gray-900">#{trainingStatus.job_id}</div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">모델</div>
                  <div className="font-medium text-gray-900">{trainingStatus.model}</div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">상태</div>
                  <div className={cn(
                    "font-medium",
                    trainingStatus.status === 'running' ? 'text-emerald-600' :
                    trainingStatus.status === 'completed' ? 'text-blue-600' :
                    trainingStatus.status === 'failed' ? 'text-red-600' : 'text-gray-600'
                  )}>
                    {trainingStatus.status}
                  </div>
                </div>
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">진행률</div>
                  <div className="font-medium text-gray-900">
                    {trainingStatus.current_epoch}/{trainingStatus.total_epochs} ({trainingStatus.progress_percent?.toFixed(1)}%)
                  </div>
                </div>
              </div>
              {trainingStatus.latest_metrics && (
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600 mb-1">최근 메트릭</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    {Object.entries(trainingStatus.latest_metrics).map(([key, value]: [string, any]) => (
                      <div key={key}>
                        <span className="text-gray-600">{key}: </span>
                        <span className="font-medium">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Phase 1: Inference Results Card */}
        {inferenceResults && (
          <div className="bg-gradient-to-br from-amber-50 to-orange-50 rounded-lg p-4 border border-amber-200">
            <h3 className="text-sm font-semibold text-gray-900 mb-3">🎯 추론 결과</h3>
            <div className="space-y-2 text-xs">
              {inferenceResults.image_path && (
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600">이미지 경로</div>
                  <div className="font-medium text-gray-900 break-all">{inferenceResults.image_path}</div>
                </div>
              )}
              {inferenceResults.predictions && inferenceResults.predictions.length > 0 && (
                <div className="bg-white rounded p-2">
                  <div className="text-gray-600 mb-2">예측 결과</div>
                  <div className="space-y-1">
                    {inferenceResults.predictions.map((pred: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-1.5 bg-gray-50 rounded">
                        <span className="font-medium">{pred.class || pred.label}</span>
                        <span className="text-amber-600 font-semibold">
                          {(pred.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-6 border-t border-gray-200">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="메시지를 입력하세요..."
            disabled={isLoading}
            className={cn(
              'flex-1 px-4 py-2.5 border border-gray-300 rounded-lg',
              'focus:outline-none focus:ring-2 focus:ring-violet-600 focus:border-transparent',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'text-sm'
            )}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className={cn(
              'px-4 py-2.5',
              'bg-violet-600 hover:bg-violet-700',
              'text-white font-semibold',
              'rounded-lg shadow-md',
              'transition-all duration-200',
              'disabled:opacity-40 disabled:cursor-not-allowed',
              'flex items-center gap-2'
            )}
          >
            <Send className="w-4 h-4" />
            {isLoading ? '전송 중...' : '전송'}
          </button>
        </div>
      </div>
    </div>
  )
}
