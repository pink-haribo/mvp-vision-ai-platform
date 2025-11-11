'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight, Info } from 'lucide-react'
import { cn } from '@/lib/utils/cn'

interface OptimizerConfigPanelProps {
  config: any
  onChange: (config: any) => void
  className?: string
}

export default function OptimizerConfigPanel({
  config,
  onChange,
  className
}: OptimizerConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const updateConfig = (field: string, value: any) => {
    onChange({
      ...config,
      [field]: value
    })
  }

  const optimizerTypes = [
    { value: 'adam', label: 'Adam', desc: '적응형 학습률, 대부분의 경우 좋은 선택' },
    { value: 'adamw', label: 'AdamW', desc: 'Weight decay가 개선된 Adam' },
    { value: 'sgd', label: 'SGD', desc: 'Momentum을 사용한 확률적 경사하강법' },
    { value: 'rmsprop', label: 'RMSprop', desc: '적응형 학습률, RNN에 적합' },
    { value: 'adagrad', label: 'Adagrad', desc: '희소 데이터에 적합한 적응형 학습률' }
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
          <h3 className="text-lg font-semibold text-white">Optimizer 설정</h3>
        </div>
        <span className="text-sm text-gray-400">{config.type.toUpperCase()}</span>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Optimizer Type */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Optimizer 유형
            </label>
            <select
              value={config.type}
              onChange={(e) => updateConfig('type', e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              {optimizerTypes.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label} - {opt.desc}
                </option>
              ))}
            </select>
          </div>

          {/* Learning Rate */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              학습률 (Learning Rate)
              <span className="ml-2 text-xs text-gray-500">권장: 0.0001 ~ 0.01</span>
            </label>
            <input
              type="number"
              value={config.learning_rate}
              onChange={(e) => updateConfig('learning_rate', parseFloat(e.target.value))}
              step="0.0001"
              min="0.000001"
              max="1"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>

          {/* Weight Decay */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Weight Decay (L2 정규화)
              <span className="ml-2 text-xs text-gray-500">권장: 0 ~ 0.1</span>
            </label>
            <input
              type="number"
              value={config.weight_decay}
              onChange={(e) => updateConfig('weight_decay', parseFloat(e.target.value))}
              step="0.001"
              min="0"
              max="1"
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>

          {/* Adam/AdamW specific */}
          {(config.type === 'adam' || config.type === 'adamw') && (
            <>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Beta 1
                  </label>
                  <input
                    type="number"
                    value={config.betas[0]}
                    onChange={(e) => updateConfig('betas', [parseFloat(e.target.value), config.betas[1]])}
                    step="0.01"
                    min="0"
                    max="0.999"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Beta 2
                  </label>
                  <input
                    type="number"
                    value={config.betas[1]}
                    onChange={(e) => updateConfig('betas', [config.betas[0], parseFloat(e.target.value)])}
                    step="0.001"
                    min="0"
                    max="0.9999"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="amsgrad"
                  checked={config.amsgrad}
                  onChange={(e) => updateConfig('amsgrad', e.target.checked)}
                  className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                />
                <label htmlFor="amsgrad" className="ml-2 text-sm text-gray-300">
                  AMSGrad 사용
                  <span className="ml-2 text-xs text-gray-500">(변동성이 큰 경우 유용)</span>
                </label>
              </div>
            </>
          )}

          {/* SGD specific */}
          {config.type === 'sgd' && (
            <>
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Momentum
                  <span className="ml-2 text-xs text-gray-500">권장: 0.9</span>
                </label>
                <input
                  type="number"
                  value={config.momentum}
                  onChange={(e) => updateConfig('momentum', parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="nesterov"
                  checked={config.nesterov}
                  onChange={(e) => updateConfig('nesterov', e.target.checked)}
                  className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                />
                <label htmlFor="nesterov" className="ml-2 text-sm text-gray-300">
                  Nesterov Momentum 사용
                  <span className="ml-2 text-xs text-gray-500">(일반적으로 성능 향상)</span>
                </label>
              </div>
            </>
          )}

          {/* RMSprop specific */}
          {config.type === 'rmsprop' && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Alpha (smoothing constant)
              </label>
              <input
                type="number"
                value={config.alpha}
                onChange={(e) => updateConfig('alpha', parseFloat(e.target.value))}
                step="0.01"
                min="0"
                max="1"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
          )}

          {/* Info Box */}
          <div className="flex items-start gap-2 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-blue-300">
              <p className="font-medium mb-1">Optimizer 선택 가이드</p>
              <ul className="list-disc list-inside space-y-1 text-xs">
                <li>일반적인 경우: AdamW (weight decay 포함)</li>
                <li>빠른 수렴: Adam</li>
                <li>더 나은 일반화: SGD with Momentum</li>
                <li>RNN/LSTM: RMSprop</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
