'use client'

import { useState, useEffect } from 'react'
import { SlidePanel } from '../SlidePanel'
import {
  TrendingUp,
  CheckCircle,
  XCircle,
  Lightbulb,
  ArrowRight,
  Settings,
  Sparkles,
  Target,
  Loader2,
  AlertCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface ModelGuideDrawerProps {
  isOpen: boolean
  onClose: () => void
  framework: string
  modelName: string
}

interface GuideData {
  model: {
    framework: string
    model_name: string
    display_name: string
    description: string
    params: string
    input_size: number
    task_types: string[]  // Changed from task_type to task_types
    tags: string[]
  }
  benchmark: Record<string, any>
  use_cases: string[]
  pros: string[]
  cons: string[]
  when_to_use: string
  when_not_to_use?: string
  alternatives: Array<{ model: string; reason: string }>
  recommended_settings: Record<string, any>
  real_world_examples?: Array<{
    title: string
    description: string
    metrics?: Record<string, string>
    link?: string
  }>
  special_features?: {
    type: string
    capabilities?: string[]
    example_prompts?: string[]
    usage_example?: Record<string, string>
    prompt_engineering_tips?: string[]
    [key: string]: any
  }
}

export default function ModelGuideDrawer({
  isOpen,
  onClose,
  framework,
  modelName,
}: ModelGuideDrawerProps) {
  const [guideData, setGuideData] = useState<GuideData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (isOpen && framework && modelName) {
      fetchGuideData()
    }
  }, [isOpen, framework, modelName])

  const fetchGuideData = async () => {
    try {
      setLoading(true)
      setError(null)

      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/models/${framework}/${modelName}/guide`
      )

      if (!response.ok) {
        throw new Error('Failed to fetch model guide')
      }

      const data = await response.json()
      setGuideData(data)
    } catch (err) {
      console.error('Error fetching guide:', err)
      setError('가이드를 불러오는데 실패했습니다')
    } finally {
      setLoading(false)
    }
  }

  const renderContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-64">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      )
    }

    if (error || !guideData) {
      return (
        <div className="p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-600">{error || '데이터를 불러올 수 없습니다'}</p>
        </div>
      )
    }

    return (
      <div className="p-6 space-y-8">
        {/* Model Header */}
        <div className="pb-6 border-b border-gray-200">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {guideData.model.display_name}
          </h2>
          <p className="text-gray-600 mb-4">{guideData.model.description}</p>
          <div className="flex flex-wrap gap-2">
            {guideData.model.tags.map((tag) => (
              <span
                key={tag}
                className="px-2 py-1 rounded-md text-xs bg-gray-100 text-gray-700"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        {/* Quick Stats */}
        {guideData.benchmark && Object.keys(guideData.benchmark).length > 0 && (
          <section>
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-blue-600" />
              Quick Stats
            </h3>
            <div className="grid grid-cols-2 gap-4">
              {Object.entries(guideData.benchmark).map(([key, value]) => (
                <div
                  key={key}
                  className="p-4 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <div className="text-xs text-gray-500 mb-1">
                    {key.replace(/_/g, ' ').toUpperCase()}
                  </div>
                  <div className="text-lg font-bold text-gray-900">{String(value)}</div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Usage Guidance */}
        <section>
          <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-600" />
            Usage Guidance
          </h3>

          {/* Pros */}
          {guideData.pros && guideData.pros.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-green-900 mb-2">✅ Pros</h4>
              <ul className="space-y-2">
                {guideData.pros.map((pro, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 shrink-0" />
                    <span>{pro}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Cons */}
          {guideData.cons && guideData.cons.length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-red-900 mb-2">❌ Cons</h4>
              <ul className="space-y-2">
                {guideData.cons.map((con, idx) => (
                  <li key={idx} className="flex items-start gap-2 text-sm text-gray-700">
                    <XCircle className="w-4 h-4 text-red-600 mt-0.5 shrink-0" />
                    <span>{con}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* When to Use */}
          <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h4 className="text-sm font-semibold text-blue-900 mb-2">💡 When to Use</h4>
            <p className="text-sm text-blue-800">{guideData.when_to_use}</p>
            {guideData.when_not_to_use && (
              <>
                <h4 className="text-sm font-semibold text-blue-900 mt-3 mb-2">
                  ⚠️ When NOT to Use
                </h4>
                <p className="text-sm text-blue-800">{guideData.when_not_to_use}</p>
              </>
            )}
          </div>
        </section>

        {/* Special Features (for YOLO-World, etc.) */}
        {guideData.special_features && (
          <section>
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-purple-600" />
              Special Features
            </h3>
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg space-y-4">
              {/* Capabilities */}
              {guideData.special_features.capabilities && (
                <div>
                  <h4 className="text-sm font-semibold text-purple-900 mb-2">
                    🚀 Capabilities
                  </h4>
                  <ul className="space-y-2">
                    {guideData.special_features.capabilities.map((capability, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2 text-sm text-purple-800"
                      >
                        <Target className="w-4 h-4 text-purple-600 mt-0.5 shrink-0" />
                        <span>{capability}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Example Prompts */}
              {guideData.special_features.example_prompts && (
                <div>
                  <h4 className="text-sm font-semibold text-purple-900 mb-2">
                    📝 Example Prompts
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {guideData.special_features.example_prompts.map((prompt, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 rounded-md text-xs bg-purple-100 text-purple-800"
                      >
                        {prompt}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Usage Example */}
              {guideData.special_features.usage_example && (
                <div>
                  <h4 className="text-sm font-semibold text-purple-900 mb-2">
                    💻 Usage Example
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(guideData.special_features.usage_example).map(
                      ([key, value]) => (
                        <div key={key} className="p-2 bg-white rounded border border-purple-200">
                          <div className="text-xs text-purple-700 mb-1">{key}</div>
                          <code className="text-xs text-gray-800">{value}</code>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

              {/* Prompt Engineering Tips */}
              {guideData.special_features.prompt_engineering_tips && (
                <div>
                  <h4 className="text-sm font-semibold text-purple-900 mb-2">
                    💡 Prompt Engineering Tips
                  </h4>
                  <ul className="space-y-1">
                    {guideData.special_features.prompt_engineering_tips.map((tip, idx) => (
                      <li key={idx} className="text-sm text-purple-800">
                        • {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Similar Models */}
        {guideData.alternatives && guideData.alternatives.length > 0 && (
          <section>
            <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
              <ArrowRight className="w-5 h-5 text-gray-600" />
              Similar Models
            </h3>
            <div className="space-y-3">
              {guideData.alternatives.map((alt, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-gray-50 rounded-lg border border-gray-200 hover:border-blue-300 transition-colors"
                >
                  <div className="font-semibold text-gray-900 mb-1">{alt.model}</div>
                  <p className="text-sm text-gray-600">{alt.reason}</p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Recommended Settings */}
        {guideData.recommended_settings &&
          Object.keys(guideData.recommended_settings).length > 0 && (
            <section>
              <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-gray-600" />
                Recommended Settings
              </h3>
              <div className="space-y-3">
                {Object.entries(guideData.recommended_settings).map(([key, value]) => (
                  <div
                    key={key}
                    className="p-4 bg-gray-50 rounded-lg border border-gray-200"
                  >
                    <div className="font-semibold text-gray-900 mb-2">
                      {key.replace(/_/g, ' ').toUpperCase()}
                    </div>
                    {typeof value === 'object' ? (
                      <div className="space-y-1 text-sm text-gray-700">
                        {Object.entries(value).map(([k, v]) => (
                          <div key={k}>
                            <span className="font-medium">{k}:</span> {String(v)}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-gray-700">{String(value)}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

        {/* Real-world Examples */}
        {guideData.real_world_examples && guideData.real_world_examples.length > 0 && (
          <section className="pb-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">🌍 Real-world Examples</h3>
            <div className="space-y-4">
              {guideData.real_world_examples.map((example, idx) => (
                <div
                  key={idx}
                  className="p-4 bg-green-50 border border-green-200 rounded-lg"
                >
                  <h4 className="font-semibold text-green-900 mb-2">{example.title}</h4>
                  <p className="text-sm text-green-800 mb-2">{example.description}</p>
                  {example.metrics && (
                    <div className="space-y-1 text-sm">
                      {Object.entries(example.metrics).map(([key, value]) => (
                        <div key={key} className="text-green-700">
                          <span className="font-medium">{key}:</span> {value}
                        </div>
                      ))}
                    </div>
                  )}
                  {example.link && (
                    <a
                      href={example.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:text-blue-800 mt-2 inline-block"
                    >
                      Learn more →
                    </a>
                  )}
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Use Cases */}
        {guideData.use_cases && guideData.use_cases.length > 0 && (
          <section className="pb-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">📋 Use Cases</h3>
            <ul className="space-y-2">
              {guideData.use_cases.map((useCase, idx) => (
                <li
                  key={idx}
                  className="flex items-start gap-2 text-sm text-gray-700 p-3 bg-gray-50 rounded-lg"
                >
                  <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" />
                  <span>{useCase}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    )
  }

  return (
    <SlidePanel
      isOpen={isOpen}
      onClose={onClose}
      title="Model Guide"
      width="xl"
    >
      {renderContent()}
    </SlidePanel>
  )
}
