'use client'

import { useState, useEffect, useRef } from 'react'
import { Send } from 'lucide-react'
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
}

export default function ChatPanel({
  sessionId,
  onSessionCreated,
  onTrainingRequested,
}: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [capabilities, setCapabilities] = useState<Capability | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

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
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
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

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900">채팅</h2>
        <p className="text-sm text-gray-600 mt-1">
          자연어로 학습 설정을 입력하세요
        </p>
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
