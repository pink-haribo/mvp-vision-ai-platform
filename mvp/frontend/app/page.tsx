'use client'

import { useState, useRef, useEffect } from 'react'
import Sidebar from '@/components/Sidebar'
import ChatPanel from '@/components/ChatPanel'
import TrainingPanel from '@/components/TrainingPanel'

export default function Home() {
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [trainingJobId, setTrainingJobId] = useState<number | null>(null)
  const [leftWidth, setLeftWidth] = useState(30) // 30% initial width
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return

      const containerRect = containerRef.current.getBoundingClientRect()
      const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100

      // Clamp between 30% (min) and 50% (max)
      const clampedWidth = Math.max(30, Math.min(50, newLeftWidth))
      setLeftWidth(clampedWidth)
    }

    const handleMouseUp = () => {
      setIsDragging(false)
    }

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging])

  const handleNewProject = () => {
    // Reset session and training job to start fresh
    setSessionId(null)
    setTrainingJobId(null)
  }

  const handleSelectSession = (selectedSessionId: number) => {
    // TODO: Load session data from backend
    console.log('Selected session:', selectedSessionId)
  }

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <Sidebar
        onNewProject={handleNewProject}
        onSelectSession={handleSelectSession}
        currentSessionId={sessionId}
      />

      {/* Main Content */}
      <main ref={containerRef} className="flex-1 flex overflow-hidden relative">
        {/* Chat Panel - Left */}
        <div style={{ width: `${leftWidth}%` }} className="border-r border-gray-200">
          <ChatPanel
            sessionId={sessionId}
            onSessionCreated={setSessionId}
            onTrainingRequested={setTrainingJobId}
          />
        </div>

        {/* Resizer */}
        <div
          onMouseDown={handleMouseDown}
          className={`w-1 bg-gray-200 hover:bg-violet-400 cursor-col-resize transition-colors relative group ${
            isDragging ? 'bg-violet-500' : ''
          }`}
        >
          <div className="absolute inset-y-0 -left-1 -right-1" />
        </div>

        {/* Training Panel - Right */}
        <div style={{ width: `${100 - leftWidth}%` }} className="flex-1">
          <TrainingPanel trainingJobId={trainingJobId} />
        </div>
      </main>
    </div>
  )
}
