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
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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
          <div className="text-center text-gray-500 mt-8">
            <p className="text-sm">대화를 시작하세요</p>
            <p className="text-xs mt-2">
              예: "ResNet50으로 10개 클래스 분류 모델을 학습하고 싶어요"
            </p>
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
