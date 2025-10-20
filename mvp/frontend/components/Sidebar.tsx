'use client'

import { useState } from 'react'
import { Plus, MessageSquare, User } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface SidebarProps {
  onNewProject?: () => void
  onSelectSession?: (sessionId: number) => void
  currentSessionId?: number | null
}

export default function Sidebar({
  onNewProject,
  onSelectSession,
  currentSessionId,
}: SidebarProps) {
  // Dummy recent sessions data
  const recentSessions = [
    { id: 1, title: 'ResNet50 Classification', timestamp: '2시간 전' },
    { id: 2, title: 'YOLO Object Detection', timestamp: '어제' },
    { id: 3, title: 'Image Segmentation', timestamp: '2일 전' },
    { id: 4, title: 'Style Transfer', timestamp: '3일 전' },
    { id: 5, title: 'Face Recognition', timestamp: '1주일 전' },
  ]

  // Dummy user data
  const user = {
    name: 'John Doe',
    email: 'john.doe@example.com',
    avatar: 'JD',
  }

  return (
    <div className="w-64 h-screen bg-gray-900 text-white flex flex-col">
      {/* Project Title */}
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-xl font-bold bg-gradient-to-r from-violet-400 to-purple-400 bg-clip-text text-transparent">
          Vision AI Platform
        </h1>
        <p className="text-xs text-gray-400 mt-1">Training Platform MVP</p>
      </div>

      {/* New Project Button */}
      <div className="px-4 py-3">
        <button
          onClick={onNewProject}
          className={cn(
            'w-full px-4 py-3',
            'bg-violet-600 hover:bg-violet-700',
            'text-white font-semibold',
            'rounded-lg',
            'transition-all duration-200',
            'flex items-center justify-center gap-2',
            'shadow-lg hover:shadow-xl'
          )}
        >
          <Plus className="w-5 h-5" />
          새 프로젝트
        </button>
      </div>

      {/* Recent Sessions */}
      <div className="flex-1 overflow-hidden flex flex-col px-4 py-2">
        <h2 className="text-sm font-semibold text-gray-400 mb-2 px-2">최근 대화</h2>
        <div className="flex-1 overflow-y-auto space-y-1">
          {recentSessions.map((session) => (
            <button
              key={session.id}
              onClick={() => onSelectSession?.(session.id)}
              className={cn(
                'w-full px-3 py-2.5 rounded-lg',
                'text-left transition-colors',
                'flex items-start gap-2',
                'group',
                currentSessionId === session.id
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-300 hover:bg-gray-800/50'
              )}
            >
              <MessageSquare className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-500 group-hover:text-violet-400" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{session.title}</p>
                <p className="text-xs text-gray-500">{session.timestamp}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* User Info */}
      <div className="border-t border-gray-800 p-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-violet-500 to-purple-600 rounded-full flex items-center justify-center font-semibold">
            {user.avatar}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold truncate">{user.name}</p>
            <p className="text-xs text-gray-400 truncate">{user.email}</p>
          </div>
        </div>
      </div>
    </div>
  )
}
