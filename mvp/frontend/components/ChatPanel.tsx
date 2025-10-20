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
  models: Array<{
    name: string
    display_name: string
    description: string
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

      // Refocus input after message is sent
      setTimeout(() => {
        inputRef.current?.focus()
      }, 0)

      // If training config is complete, create training job
      if (data.parsed_intent?.status === 'complete') {
        console.log('Training config ready:', data.parsed_intent.config)
        await createTrainingJob(data.session_id, data.parsed_intent.config)
      }
    } catch (error) {
      console.error('Error sending message:', error)
      alert('메시지 전송 실패')
    } finally {
      setIsLoading(false)
    }
  }

  const createTrainingJob = async (sessionId: number, config: any) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/training/jobs`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          config: config,
        }),
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
                        </div>
                        <div className="text-gray-600 mt-0.5">{model.description}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Configurable Parameters */}
                <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-4 border border-blue-200">
                  <h3 className="text-sm font-semibold text-gray-900 mb-2">
                    설정 가능한 파라미터
                  </h3>
                  <div className="space-y-2">
                    {capabilities.parameters.map((param) => (
                      <div key={param.name} className="text-xs bg-white p-2 rounded">
                        <div className="font-medium text-gray-900">
                          {param.display_name}
                          {param.required && (
                            <span className="ml-2 text-red-600">* 필수</span>
                          )}
                          {!param.required && param.default !== null && (
                            <span className="ml-2 text-gray-500">
                              (기본값: {param.default})
                            </span>
                          )}
                        </div>
                        <div className="text-gray-600 mt-0.5">{param.description}</div>
                        {param.type === 'integer' && param.min && param.max && (
                          <div className="text-gray-500 mt-0.5">
                            범위: {param.min} ~ {param.max}
                          </div>
                        )}
                        {param.type === 'float' && param.min && param.max && (
                          <div className="text-gray-500 mt-0.5">
                            범위: {param.min} ~ {param.max}
                          </div>
                        )}
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
