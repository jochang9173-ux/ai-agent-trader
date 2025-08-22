'use client'

import React, { useState, useCallback, useEffect } from 'react'
import { Calendar, TrendingUp, FileText, BarChart3, AlertTriangle, MessageSquare, Lightbulb } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Textarea } from '@/components/ui/textarea'
import { DayAnalysisResponse, TrendAnalysis, RetrospectiveAnalysis, TechnicalEvent } from '@/types'

interface DayAnalysisProps {
  runId: string
  onDateSelect: (date: string) => void
}

interface DailyFeedbackSectionProps {
  date: string
}

interface DailyImprovementResponse {
  analysis: string
  suggestions: string[]
}

interface DayAnalysisState {
  selectedDate: string | null
  analysis: DayAnalysisResponse | null
  availableDates: string[]
  isLoading: boolean
  isLoadingDates: boolean
  error: string | null
}

/**
 * Clean markdown formatting from text
 */
const cleanMarkdown = (text: string): string => {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markdown **text**
    .replace(/\*(.*?)\*/g, '$1')     // Remove italic markdown *text*
    .replace(/`(.*?)`/g, '$1')       // Remove code markdown `text`
    .trim()
}

/**
 * Day Analysis Component - Provides detailed analysis of specific trading days
 * Allows users to select dates and view comprehensive analysis including LLM decisions
 */
function DayAnalysis({ runId, onDateSelect }: DayAnalysisProps) {
  const [state, setState] = useState<DayAnalysisState>({
    selectedDate: null,
    analysis: null,
    availableDates: [],
    isLoading: false,
    isLoadingDates: true,
    error: null
  })

  // Fetch available dates when component mounts or runId changes
  useEffect(() => {
    const fetchAvailableDates = async () => {
      setState(prev => ({ ...prev, isLoadingDates: true, error: null }))
      
      try {
        const response = await fetch(`/api/v1/backtest/available-dates/${runId}`)
        if (!response.ok) {
          throw new Error(`Failed to fetch available dates: ${response.statusText}`)
        }
        
        const data = await response.json()
        setState(prev => ({ 
          ...prev, 
          availableDates: data.dates || [],
          isLoadingDates: false 
        }))
      } catch (error) {
        setState(prev => ({ 
          ...prev, 
          error: error instanceof Error ? error.message : 'Failed to load available dates',
          isLoadingDates: false,
          availableDates: []
        }))
      }
    }

    if (runId) {
      fetchAvailableDates()
    }
  }, [runId])

  // Handle date selection and fetch analysis
  const handleDateSelect = useCallback(async (date: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null, selectedDate: date }))
    onDateSelect(date)

    try {
      const response = await fetch(`/api/v1/backtest/analysis/day/${runId}?date=${date}&include_retrospective=false`)
      if (!response.ok) {
        throw new Error(`Failed to fetch analysis: ${response.statusText}`)
      }
      
      const analysisData: DayAnalysisResponse = await response.json()
      setState(prev => ({ 
        ...prev, 
        analysis: analysisData, 
        isLoading: false 
      }))
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Unknown error occurred',
        isLoading: false 
      }))
    }
  }, [runId, onDateSelect])

  // Format date for display
  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleDateString('zh-TW', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long'
    })
  }

  // Get trend color based on trend type
  const getTrendColor = (trend: string): string => {
    switch (trend) {
      case 'BULLISH': return 'text-green-600'
      case 'BEARISH': return 'text-red-600'
      case 'SIDEWAYS': return 'text-gray-600'
      default: return 'text-gray-600'
    }
  }

  // Get badge variant based on impact
  const getImpactVariant = (impact: string): "default" | "secondary" | "destructive" | "outline" => {
    switch (impact) {
      case 'POSITIVE': return 'default'
      case 'NEGATIVE': return 'destructive'
      case 'NEUTRAL': return 'secondary'
      default: return 'outline'
    }
  }

  // Translate technical event types to Chinese
  const translateEventType = (eventType: string): string => {
    const translations: Record<string, string> = {
      // 布林帶相關
      'BB_UPPER_TOUCH': '觸及布林上軌',
      'BB_LOWER_TOUCH': '觸及布林下軌',
      'BB_SQUEEZE': '布林帶收縮',
      'BB_EXPANSION': '布林帶擴張',
      
      // 移動平均線相關
      'MA_GOLDEN_CROSS': '均線黃金交叉',
      'MA_DEATH_CROSS': '均線死亡交叉',
      'MA_SUPPORT': '均線支撐',
      'MA_RESISTANCE': '均線阻力',
      
      // MACD相關
      'MACD_GOLDEN_CROSS': 'MACD黃金交叉',
      'MACD_DEATH_CROSS': 'MACD死亡交叉',
      'MACD_DIVERGENCE': 'MACD背離',
      
      // RSI相關
      'RSI_OVERSOLD': 'RSI超賣',
      'RSI_OVERBOUGHT': 'RSI超買',
      'RSI_DIVERGENCE': 'RSI背離',
      
      // 成交量相關
      'VOLUME_SPIKE': '成交量暴增',
      'VOLUME_DRY_UP': '成交量萎縮',
      'VOLUME_BREAKOUT': '放量突破',
      'HIGH_VOLUME': '成交量爆量',
      'VOLUME_EXPLOSION': '成交量爆量',
      
      // 趨勢相關
      'TREND_TURN_BULLISH': '趨勢轉多',
      'TREND_TURN_BEARISH': '趨勢轉空',
      'TREND_ACCELERATION': '趨勢加速',
      'TREND_WEAKNESS': '趨勢疲弱',
      
      // 動量相關
      'MOMENTUM_SHIFT': '動量轉變',
      'MOMENTUM_DIVERGENCE': '動量背離',
      
      // 其他
      'GAP_UP': '向上跳空',
      'GAP_DOWN': '向下跳空',
      'HIGH_VOLATILITY': '高波動',
      'LOW_VOLATILITY': '低波動',
      
      // 未知或其他事件類型的默認處理
      'unknown': '技術事件',
      'UNKNOWN': '技術事件',
      'OTHER': '其他技術信號'
    }
    
    return translations[eventType] || `技術事件: ${eventType}`
  }

  // Translate severity levels to Chinese
  const translateSeverity = (severity: string): string => {
    const translations: Record<string, string> = {
      'high': '高',
      'medium': '中',
      'low': '低',
      'very_high': '極高',
      'very_low': '極低'
    }
    
    return translations[severity] || severity
  }

  return (
    <div className="space-y-6">
      {/* Date Selection Header */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            📅 交易日誌探索
          </CardTitle>
          <CardDescription>
            選擇一個交易日，我們一起回顧那天的決策過程，看看有什麼值得討論的地方！
          </CardDescription>
        </CardHeader>
        <CardContent>
          {state.isLoadingDates ? (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 mx-auto"></div>
              <p className="mt-2 text-sm text-gray-600">載入可用日期中...</p>
            </div>
          ) : state.availableDates.length === 0 ? (
            <div className="text-center py-4 text-gray-500">
              沒有找到可用的分析日期
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {state.availableDates.map((date: string) => (
                <Button
                  key={date}
                  variant={state.selectedDate === date ? "default" : "outline"}
                  size="sm"
                  onClick={() => handleDateSelect(date)}
                  disabled={state.isLoading}
                  className="text-xs"
                >
                  {new Date(date).toLocaleDateString('zh-TW', { 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </Button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error State */}
      {state.error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{state.error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {state.isLoading && (
        <div className="space-y-4">
          <div className="h-32 w-full bg-gray-200 animate-pulse rounded-lg" />
          <div className="h-48 w-full bg-gray-200 animate-pulse rounded-lg" />
          <div className="h-64 w-full bg-gray-200 animate-pulse rounded-lg" />
        </div>
      )}

      {/* Analysis Results */}
      {state.analysis && !state.isLoading && (
        <div className="space-y-6">
          {/* Technical Events */}
          {state.analysis.historical_data.technical_events.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" />
                  技術事件
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {/* Original technical events */}
                  {state.analysis.historical_data.technical_events.map((event, index) => (
                    <div key={`original-${index}`} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <div className="font-medium">{translateEventType(event.event_type)}</div>
                        <div className="text-sm text-gray-600">{event.description}</div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className={event.severity === 'high' ? 'text-red-600' : 'text-yellow-600'}>
                          {translateSeverity(event.severity)}
                        </Badge>
                      </div>
                    </div>
                  ))}
                  
                  {/* No events message */}
                  {state.analysis.historical_data.technical_events.length === 0 && (
                    <div className="text-center py-8 text-gray-500">
                      當日無重要技術事件
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* LLM Analysis */}
          {state.analysis.historical_data.llm_decision && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-purple-500 flex items-center justify-center">
                    <span className="text-white text-xs font-bold">AI</span>
                  </div>
                  🧠 AI的當日思考過程
                </CardTitle>
                <CardDescription>
                  讓我們看看AI那天是怎麼想的，你覺得它的推理合理嗎？
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">決策類型</div>
                      <div className="text-lg font-bold">
                        {(() => {
                          const decisionType = state.analysis.historical_data.llm_decision.decision_type
                          const strategyState = state.analysis.historical_data.strategy_state
                          
                          // 檢查是否有持倉信息 - 根據實際數據結構
                          const hasPosition = strategyState?.shares > 0 || 
                                            strategyState?.position === 'long' || 
                                            strategyState?.position === 'short'
                          
                          if (decisionType === 'BUY') {
                            return '📈 買入'
                          } else if (decisionType === 'SELL') {
                            return '📉 賣出'
                          } else if (decisionType === 'HOLD') {
                            // 根據持倉狀態決定顯示內容
                            if (hasPosition) {
                              return '⏸️ 持有'
                            } else {
                              return '💤 空倉觀望'
                            }
                          } else {
                            return '⏸️ 觀望'
                          }
                        })()}
                      </div>
                    </div>
                  </div>
                  
                  {state.analysis.historical_data.llm_decision.reasoning && (
                    <div>
                      <div className="text-sm text-gray-600 mb-2">分析推理</div>
                      <div className="text-sm bg-gray-50 p-3 rounded-lg whitespace-pre-line">
                        {state.analysis.historical_data.llm_decision.reasoning}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Daily Decision Improvement */}
          {state.selectedDate && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-5 w-5" />
                  💬 策略討論室
                </CardTitle>
                <CardDescription>
                  與AI助手一起探討交易決策，分享你的見解並獲得策略優化建議
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DailyFeedbackSection 
                  date={state.selectedDate}
                />
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* No Date Selected State */}
      {!state.selectedDate && !state.isLoading && (
        <Card>
          <CardContent className="text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-100 to-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Calendar className="h-8 w-8 text-blue-500" />
            </div>
            <h3 className="text-lg font-medium text-gray-700 mb-2">🚀 準備開始我們的策略討論吧！</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              從上方選擇一個交易日，我會告訴你那天發生了什麼，然後我們可以一起聊聊策略優化的想法 💭
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

const DailyFeedbackSection: React.FC<DailyFeedbackSectionProps> = ({ date }) => {
  const [feedback, setFeedback] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<DailyImprovementResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmitFeedback = async () => {
    if (!feedback.trim()) {
      setError('請告訴我你的想法！')
      return
    }

    setIsAnalyzing(true)
    setError(null)

    try {
      const response = await fetch('http://localhost:8000/api/v1/daily/daily-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          feedback: feedback.trim(),
          date: date
        })
      })

      if (!response.ok) {
        throw new Error(`討論過程中發生問題: ${response.status}`)
      }

      const data: DailyImprovementResponse = await response.json()
      setResult(data)
      
    } catch (err) {
      console.error('Daily feedback error:', err)
      setError(err instanceof Error ? err.message : '討論過程中發生錯誤')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setFeedback('')
    setResult(null)
    setError(null)
  }

  return (
    <div className="space-y-4">
      {/* Interactive Header */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center">
            <span className="text-white text-sm font-bold">AI</span>
          </div>
          <h4 className="font-medium text-blue-900">策略討論助手</h4>
        </div>
        <p className="text-sm text-blue-700">
          我想聽聽你對 <span className="font-semibold">{date}</span> 這天決策的看法！我們一起來探討交易策略的優化方向 🤔
        </p>
      </div>

      {/* Input Section */}
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium text-gray-700 mb-2 block flex items-center gap-2">
            💬 你的想法是...
          </label>
          <Textarea
            placeholder="嗨！告訴我你的想法吧... 比如：「我覺得這天不該賣出，因為...」或者「我同意這個決策，但是...」"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            rows={4}
            className="w-full border-2 border-gray-200 focus:border-blue-400 transition-colors"
            disabled={isAnalyzing}
          />
        </div>
        
        <div className="flex gap-2">
          <Button 
            onClick={handleSubmitFeedback}
            disabled={isAnalyzing || !feedback.trim()}
            className="flex items-center gap-2 bg-blue-500 hover:bg-blue-600"
          >
            <MessageSquare className="h-4 w-4" />
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                思考中...
              </>
            ) : (
              '開始討論 💭'
            )}
          </Button>
          
          {result && (
            <Button 
              variant="outline" 
              onClick={handleReset}
              disabled={isAnalyzing}
              className="border-blue-300 text-blue-600 hover:bg-blue-50"
            >
              🔄 重新討論
            </Button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert className="bg-red-50 border-red-200">
          <AlertTriangle className="h-4 w-4 text-red-500" />
          <AlertDescription className="text-red-700">{error}</AlertDescription>
        </Alert>
      )}

      {/* Results Display - Chat Style */}
      {result && (
        <div className="space-y-4 border-t pt-4">
          {/* AI Response */}
          <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 flex items-center justify-center flex-shrink-0">
                <span className="text-white text-sm font-bold">AI</span>
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-medium text-gray-900">策略分析助手</span>
                  <Badge variant="secondary" className="text-xs">剛剛</Badge>
                </div>
                
                {/* Analysis as conversation */}
                <div className="bg-gray-50 p-3 rounded-lg text-sm text-gray-700 mb-3">
                  <div className="flex items-start gap-2">
                    <BarChart3 className="h-4 w-4 mt-0.5 text-blue-500 flex-shrink-0" />
                    <div>
                      <div className="font-medium text-gray-900 mb-1">我的看法：</div>
                      <div className="whitespace-pre-wrap">{cleanMarkdown(result.analysis)}</div>
                    </div>
                  </div>
                </div>

                {/* Suggestions as strategy file modifications */}
                {result.suggestions.length > 0 && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-sm font-medium text-gray-900">
                      <FileText className="h-4 w-4 text-green-500" />
                      📝 策略文件修改建議（traditional_strategy.md）：
                    </div>
                    {result.suggestions.map((suggestion, index) => {
                      // 分離標題和詳細內容
                      const lines = suggestion.split('\n')
                      const title = lines[0] || suggestion
                      const details = lines.slice(1).join('\n').trim()
                      
                      return (
                        <div key={index} className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 p-4 rounded-lg text-sm">
                          <div className="flex items-start gap-3">
                            <span className="text-green-600 font-bold text-xs bg-green-100 px-2 py-1 rounded-full flex-shrink-0">
                              修改 {index + 1}
                            </span>
                            <div className="text-gray-700 flex-1">
                              <div className="flex items-center gap-2 mb-3">
                                <div className="font-mono text-xs text-green-800 bg-green-100 px-2 py-1 rounded">
                                  📄 traditional_strategy.md
                                </div>
                                <div className="text-xs text-green-600 font-medium">策略文件修改</div>
                              </div>
                              
                              {/* 標題 */}
                              <div className="font-semibold text-gray-800 mb-2">
                                {cleanMarkdown(title)}
                              </div>
                              
                              {/* 詳細內容 - 如果有的話 */}
                              {details && (
                                <div className="text-gray-600 text-xs leading-relaxed bg-white bg-opacity-50 p-3 rounded border-l-2 border-green-300">
                                  <div className="whitespace-pre-wrap">{cleanMarkdown(details)}</div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}

                {/* Encourage further discussion */}
                <div className="mt-4 pt-3 border-t border-gray-100">
                  <p className="text-xs text-gray-500 italic">
                    � 這些建議可以直接應用到策略文件中！有其他優化想法嗎？點擊「重新討論」繼續完善我們的交易策略！
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
export default DayAnalysis
