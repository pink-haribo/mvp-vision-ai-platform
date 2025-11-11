'use client'

import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { cn } from '@/lib/utils'

interface AugmentationConfigPanelProps {
  config: any
  onChange: (config: any) => void
  className?: string
}

export default function AugmentationConfigPanel({
  config,
  onChange,
  className
}: AugmentationConfigPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  const updateConfig = (field: string, value: any) => {
    onChange({
      ...config,
      [field]: value
    })
  }

  const enabledCount = [
    config.random_flip,
    config.random_rotation,
    config.random_crop,
    config.color_jitter,
    config.random_erasing,
    config.mixup,
    config.cutmix
  ].filter(Boolean).length

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
          <h3 className="text-lg font-semibold text-white">Data Augmentation</h3>
        </div>
        <span className="text-sm text-gray-400">{enabledCount}개 활성화</span>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Enable/Disable Toggle */}
          <div className="flex items-center p-3 bg-gray-700/50 rounded-lg">
            <input
              type="checkbox"
              id="aug-enabled"
              checked={config.enabled}
              onChange={(e) => updateConfig('enabled', e.target.checked)}
              className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
            />
            <label htmlFor="aug-enabled" className="ml-2 text-sm font-medium text-gray-300">
              Augmentation 활성화
            </label>
          </div>

          {config.enabled && (
            <>
              {/* Random Flip */}
              <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="random-flip"
                    checked={config.random_flip}
                    onChange={(e) => updateConfig('random_flip', e.target.checked)}
                    className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                  />
                  <label htmlFor="random-flip" className="ml-2 text-sm text-gray-300">
                    Random Horizontal Flip
                  </label>
                </div>
                {config.random_flip && (
                  <input
                    type="number"
                    value={config.random_flip_prob}
                    onChange={(e) => updateConfig('random_flip_prob', parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    max="1"
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                    placeholder="확률"
                  />
                )}
              </div>

              {/* Random Rotation */}
              <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="random-rotation"
                    checked={config.random_rotation}
                    onChange={(e) => updateConfig('random_rotation', e.target.checked)}
                    className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                  />
                  <label htmlFor="random-rotation" className="ml-2 text-sm text-gray-300">
                    Random Rotation
                  </label>
                </div>
                {config.random_rotation && (
                  <input
                    type="number"
                    value={config.rotation_degrees}
                    onChange={(e) => updateConfig('rotation_degrees', parseInt(e.target.value))}
                    min="0"
                    max="180"
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                    placeholder="각도"
                  />
                )}
              </div>

              {/* Color Jitter */}
              <div className="p-3 bg-gray-700/30 rounded-lg space-y-2">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="color-jitter"
                    checked={config.color_jitter}
                    onChange={(e) => updateConfig('color_jitter', e.target.checked)}
                    className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                  />
                  <label htmlFor="color-jitter" className="ml-2 text-sm text-gray-300">
                    Color Jitter
                  </label>
                </div>
                {config.color_jitter && (
                  <div className="grid grid-cols-2 gap-2 pl-6">
                    <div>
                      <label className="text-xs text-gray-400">Brightness</label>
                      <input
                        type="number"
                        value={config.brightness}
                        onChange={(e) => updateConfig('brightness', parseFloat(e.target.value))}
                        step="0.1"
                        min="0"
                        max="1"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-gray-400">Contrast</label>
                      <input
                        type="number"
                        value={config.contrast}
                        onChange={(e) => updateConfig('contrast', parseFloat(e.target.value))}
                        step="0.1"
                        min="0"
                        max="1"
                        className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Mixup */}
              <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="mixup"
                    checked={config.mixup}
                    onChange={(e) => updateConfig('mixup', e.target.checked)}
                    className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                  />
                  <label htmlFor="mixup" className="ml-2 text-sm text-gray-300">
                    Mixup
                    <span className="ml-2 text-xs text-gray-500">(이미지 혼합)</span>
                  </label>
                </div>
                {config.mixup && (
                  <input
                    type="number"
                    value={config.mixup_alpha}
                    onChange={(e) => updateConfig('mixup_alpha', parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                    placeholder="Alpha"
                  />
                )}
              </div>

              {/* CutMix */}
              <div className="flex items-center justify-between p-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="cutmix"
                    checked={config.cutmix}
                    onChange={(e) => updateConfig('cutmix', e.target.checked)}
                    className="w-4 h-4 text-violet-600 bg-gray-700 border-gray-600 rounded focus:ring-violet-500"
                  />
                  <label htmlFor="cutmix" className="ml-2 text-sm text-gray-300">
                    CutMix
                    <span className="ml-2 text-xs text-gray-500">(영역 혼합)</span>
                  </label>
                </div>
                {config.cutmix && (
                  <input
                    type="number"
                    value={config.cutmix_alpha}
                    onChange={(e) => updateConfig('cutmix_alpha', parseFloat(e.target.value))}
                    step="0.1"
                    min="0"
                    className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
                    placeholder="Alpha"
                  />
                )}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
