// 基本交易類型
export interface TradingSignal {
  timestamp: string
  signal_type: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  price: number
  reason: string
  metadata?: Record<string, any>
}

// 股票數據類型
export interface StockData {
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  // 技術指標
  ma_5?: number
  ma_10?: number
  ma_20?: number
  rsi?: number
  macd?: number
  macd_signal?: number
  macd_histogram?: number
  // 布林帶指標
  bb_upper?: number
  bb_lower?: number
  bb_middle?: number
  // 向前兼容
  bollinger_upper?: number
  bollinger_lower?: number
}

// LLM 決策日誌
export interface LLMDecisionLog {
  timestamp: string  // 時間戳
  decision: {
    action?: 'BUY' | 'SELL' | 'HOLD'
    confidence?: number
    reasoning?: string
    risk_level?: 'low' | 'medium' | 'high'
    expected_outcome?: string
  }
  reasoning: string  // 決策推理
  events: Array<{
    type: string
    description: string
    strength: 'low' | 'medium' | 'high'
  }>
  action: string    // 行動類型 (如 "THINK")
  confidence: number
  price: number
  // 兼容舊格式
  date?: string
}

// 回測結果類型
export interface BacktestResult {
  trades: any[]
  performance: any
  stock_data: StockData[]
  signals: TradingSignal[]
  llm_decisions: LLMDecisionLog[]
  statistics: {
    total_trades: number
    win_rate: number
    total_return: number
    max_drawdown: number
    final_value?: number
    total_realized_pnl?: number
    cumulative_trade_return_rate?: number
  }
}

// 技術事件類型
export interface TechnicalEvent {
  type: string
  description: string
  significance: number
  impact: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL'
}

// 趨勢分析類型
export interface TrendAnalysis {
  primary_trend: 'BULLISH' | 'BEARISH' | 'SIDEWAYS'
  trend_strength: number
  trend_duration: number
  momentum_indicators: Record<string, any>
  support_resistance: {
    support_levels: number[]
    resistance_levels: number[]
  }
}

// 回顧性分析類型
export interface RetrospectiveAnalysis {
  summary: string
  decision_quality: {
    score: number
    reasoning: string
    alternatives: string[]
  }
  market_context: {
    market_conditions: string
    volatility_assessment: string
    key_factors: string[]
  }
  performance_impact: {
    immediate_impact: string
    potential_outcomes: string
    risk_assessment: string
  }
  lessons_learned: string[]
  recommendations: string[]
}

// 日分析響應類型
// 日分析響應類型（與後端API匹配）
export interface DayAnalysisResponse {
  historical_data: {
    date: string
    symbol: string
    price: number
    daily_return?: number
    volume?: number
    market_data?: {
      open?: number
      high?: number
      low?: number
      close: number
      volume?: number
    }
    trend_analysis?: {
      short_term?: string
      medium_term?: string
      long_term?: string
      trend_strength?: number
      confidence?: number
    }
    comprehensive_technical_analysis?: {
      date?: string
      price_action?: Record<string, any>
      moving_averages?: Record<string, any>
      volume_analysis?: Record<string, any>
      volatility_analysis?: Record<string, any>
      momentum_indicators?: Record<string, any>
      support_resistance?: Record<string, any>
      trend_analysis?: Record<string, any>
      market_regime?: Record<string, any>
      bollinger_analysis?: Record<string, any>
      macd_analysis?: Record<string, any>
    }
    technical_events: Array<{
      event_type: string
      severity: string
      description: string
      technical_data?: Record<string, any>
    }>
    llm_decision?: {
      decision_made: boolean
      decision_type?: string
      confidence?: number
      reasoning?: string
      risk_level?: string
    }
    strategy_state?: Record<string, any>
  }
  retrospective_analysis?: {
    llm_commentary: string
    decision_quality_score?: number
    alternative_perspective?: string
    lessons_learned?: string
  }
}
