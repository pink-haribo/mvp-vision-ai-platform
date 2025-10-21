'use client'

import { useState, useRef, useEffect } from 'react'
import Sidebar, { NavigationView } from '@/components/Sidebar'
import ChatPanel from '@/components/ChatPanel'
import TrainingPanel from '@/components/TrainingPanel'
import ProjectList from '@/components/ProjectList'
import ProjectDetail from '@/components/ProjectDetail'

export default function Home() {
  const [currentView, setCurrentView] = useState<NavigationView>('chat')
  const [sessionId, setSessionId] = useState<number | null>(null)
  const [trainingJobId, setTrainingJobId] = useState<number | null>(null)
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null)
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

  const handleNavigationChange = (view: NavigationView) => {
    setCurrentView(view)
    // Reset selections when switching views
    if (view === 'projects') {
      setSelectedProjectId(null)
    }
  }

  const handleProjectSelect = (projectId: number) => {
    setSelectedProjectId(projectId)
  }

  const handleProjectBack = () => {
    setSelectedProjectId(null)
  }

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <Sidebar
        onNewProject={handleNewProject}
        onSelectSession={handleSelectSession}
        currentSessionId={sessionId}
        onNavigationChange={handleNavigationChange}
        currentView={currentView}
      />

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden relative">
        {/* Chat View (Chat + Training panels with resizer) */}
        {currentView === 'chat' && (
          <>
            {/* Chat Panel - Left */}
            <div
              ref={containerRef}
              style={{ width: `${leftWidth}%` }}
              className="border-r border-gray-200"
            >
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
          </>
        )}

        {/* Projects View */}
        {currentView === 'projects' && (
          <div className="flex-1 flex overflow-hidden">
            {/* Project List - Left (30% width) */}
            <div className="w-[30%] border-r border-gray-200">
              <ProjectList onProjectSelect={handleProjectSelect} />
            </div>

            {/* Project Detail - Right (70% width) */}
            <div className="flex-1">
              {selectedProjectId ? (
                <ProjectDetail projectId={selectedProjectId} onBack={handleProjectBack} />
              ) : (
                <div className="h-full flex items-center justify-center bg-gray-50">
                  <div className="text-center text-gray-500">
                    <p className="text-sm">프로젝트를 선택하세요</p>
                    <p className="text-xs mt-1">왼쪽 목록에서 프로젝트를 클릭하면 상세 정보가 표시됩니다</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
