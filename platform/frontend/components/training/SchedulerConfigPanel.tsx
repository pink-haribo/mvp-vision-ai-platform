'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface SchedulerConfigPanelProps {
  config: any
  onChange: (config: any) => void
  className?: string
}

export default function SchedulerConfigPanel({
  config,
  onChange,
  className
}: SchedulerConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const updateConfig = (field: string, value: any) => {
    onChange({
      ...config,
      [field]: value
    })
  }

  const schedulerTypes = [
    { value: 'none', label: '사용 안 함' },
    { value: 'step', label: 'Step LR' },
    { value: 'cosine', label: 'Cosine Annealing' },
    { value: 'reduce_on_plateau', label: 'Reduce on Plateau' },
    { value: 'one_cycle', label: 'One Cycle LR' }
  ]

  return (
    <div className={cn("bg-gray-800 rounded-lg p-4", className)}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-gray-400" />
          )}
          <h3 className="text-lg font-semibold text-white">Learning Rate Scheduler</h3>
        </div>
        <span className="text-sm text-gray-400">
          {config.type === 'none' ? '없음' : schedulerTypes.find(s => s.value === config.type)?.label}
        </span>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Scheduler Type */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Scheduler 유형
            </label>
            <select
              value={config.type}
              onChange={(e) => updateConfig('type', e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              {schedulerTypes.map((sched) => (
                <option key={sched.value} value={sched.value}>
                  {sched.label}
                </option>
              ))}
            </select>
          </div>

          {/* StepLR */}
          {config.type === 'step' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Step Size (epochs)
                </label>
                <input
                  type="number"
                  value={config.step_size}
                  onChange={(e) => updateConfig('step_size', parseInt(e.target.value))}
                  min="1"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Gamma
                </label>
                <input
                  type="number"
                  value={config.gamma}
                  onChange={(e) => updateConfig('gamma', parseFloat(e.target.value))}
                  step="0.1"
                  min="0.01"
                  max="1"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
            </div>
          )}

          {/* Cosine Annealing */}
          {config.type === 'cosine' && (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  T_max (epochs)
                </label>
                <input
                  type="number"
                  value={config.T_max}
                  onChange={(e) => updateConfig('T_max', parseInt(e.target.value))}
                  min="1"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Min LR
                </label>
                <input
                  type="number"
                  value={config.eta_min}
                  onChange={(e) => updateConfig('eta_min', parseFloat(e.target.value))}
                  step="0.000001"
                  min="0"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
            </div>
          )}

          {/* Reduce on Plateau */}
          {config.type === 'reduce_on_plateau' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Patience (epochs)
                  </label>
                  <input
                    type="number"
                    value={config.patience}
                    onChange={(e) => updateConfig('patience', parseInt(e.target.value))}
                    min="1"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Factor
                  </label>
                  <input
                    type="number"
                    value={config.factor}
                    onChange={(e) => updateConfig('factor', parseFloat(e.target.value))}
                    step="0.1"
                    min="0.01"
                    max="0.9"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Warmup (for all schedulers except none) */}
          {config.type !== 'none' && (
            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-sm font-medium text-gray-300 mb-3">Warmup 설정</h4>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Warmup Epochs
                  </label>
                  <input
                    type="number"
                    value={config.warmup_epochs}
                    onChange={(e) => updateConfig('warmup_epochs', parseInt(e.target.value))}
                    min="0"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">
                    Warmup LR
                  </label>
                  <input
                    type="number"
                    value={config.warmup_lr}
                    onChange={(e) => updateConfig('warmup_lr', parseFloat(e.target.value))}
                    step="0.000001"
                    min="0"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
