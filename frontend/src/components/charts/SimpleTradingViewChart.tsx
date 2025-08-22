'use client'

import React, { useEffect, useRef } from 'react'
import { createChart, Time } from 'lightweight-charts'
import { StockData, TradingSignal, LLMDecisionLog } from '@/types'

interface SimpleTradingViewChartProps {
  /** 股票價格數據 */
  stockData: StockData[]
  /** 交易信號數據 */
  signals?: TradingSignal[]
  /** LLM 決策記錄 */
  llmDecisions?: LLMDecisionLog[]
  /** 圖表高度 */
  height?: number
  /** 是否顯示成交量 */
  showVolume?: boolean
  /** 是否顯示交易信號 */
  showSignals?: boolean
  /** 是否顯示移動平均線 */
  showMA?: boolean
  /** 移動平均線週期 */
  maPeriods?: number[]
  /** 是否顯示RSI */
  showRSI?: boolean
  /** 是否顯示布林帶 */
  showBB?: boolean
  /** 是否顯示MACD */
  showMACD?: boolean
}

/**
 * 簡化版 TradingView Lightweight Charts 組件
 * 專注於K線圖表，避免複雜的型態問題
 */
export function SimpleTradingViewChart({
  stockData,
  signals = [],
  llmDecisions = [],
  height = 400,
  showVolume = true,
  showSignals = false,
  showMA = false,
  maPeriods = [10, 20],
  showRSI = false,
  showBB = false,
  showMACD = false,
}: SimpleTradingViewChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)

  // 在組件頂部添加數據驗證
  const validStockData = React.useMemo(() => {
    if (!stockData || !Array.isArray(stockData)) {
      console.warn('Invalid stock data:', stockData)
      return []
    }
    
    return stockData.filter(item => {
      // 基本數據驗證
      if (!item || typeof item !== 'object') return false
      if (!item.timestamp) return false
      if (typeof item.open !== 'number' || isNaN(item.open) || !isFinite(item.open)) return false
      if (typeof item.high !== 'number' || isNaN(item.high) || !isFinite(item.high)) return false
      if (typeof item.low !== 'number' || isNaN(item.low) || !isFinite(item.low)) return false
      if (typeof item.close !== 'number' || isNaN(item.close) || !isFinite(item.close)) return false
      if (typeof item.volume !== 'number' || isNaN(item.volume) || !isFinite(item.volume) || item.volume < 0) return false
      
      // OHLC 邏輯驗證
      if (item.high < item.low || item.high < item.open || item.high < item.close) return false
      if (item.low > item.open || item.low > item.close) return false
      
      return true
    })
  }, [stockData])

  console.log('Original data count:', stockData?.length || 0)
  console.log('Valid data count:', validStockData.length)
  if (validStockData.length > 0) {
    console.log('Sample valid data:', validStockData[0])
  }

  // 統一的時間轉換函數
  const convertTimestamp = (timestamp: string): number => {
    // 處理不同的時間格式
    let date: Date
    
    if (timestamp.includes('T')) {
      // ISO 格式: "2024-01-15T00:00:00" 或 "2024-01-15T00:00:00.000Z"
      date = new Date(timestamp)
    } else if (timestamp.includes('-')) {
      // 日期格式: "2024-01-15"
      date = new Date(timestamp + 'T00:00:00.000Z')
    } else {
      // 其他格式，嘗試直接解析
      date = new Date(timestamp)
    }
    
    // 確保日期有效
    if (isNaN(date.getTime())) {
      console.warn('無效的時間格式:', timestamp)
      return Math.floor(Date.now() / 1000)
    }
    
    // 轉換為 TradingView 所需的 Unix 時間戳（秒）
    const unixTimestamp = Math.floor(date.getTime() / 1000)
    
    // 添加調試信息（只在開發環境）
    if (process.env.NODE_ENV === 'development') {
      console.log(`時間轉換: ${timestamp} -> ${date.toISOString()} -> ${unixTimestamp}`)
    }
    
    return unixTimestamp
  }

  useEffect(() => {
    if (!chartContainerRef.current || !validStockData.length) {
      console.warn('Chart container or data not available')
      return
    }

    console.log('Creating chart with', validStockData.length, 'data points')

    // 計算所需的圖表高度分配
    let mainChartHeight = height
    let subChartsCount = 0
    if (showRSI) subChartsCount++
    if (showMACD) subChartsCount++
    
    // 如果有子圖表，主圖佔70%，子圖表平分剩餘空間
    if (subChartsCount > 0) {
      mainChartHeight = Math.floor(height * 0.7)
    }

    // 創建主圖表容器
    const mainChartContainer = document.createElement('div')
    mainChartContainer.style.height = `${mainChartHeight}px`
    chartContainerRef.current.innerHTML = ''
    chartContainerRef.current.appendChild(mainChartContainer)

    // 創建主圖表
    const chart = createChart(mainChartContainer, {
      width: chartContainerRef.current.clientWidth,
      height: mainChartHeight,
      layout: {
        backgroundColor: '#ffffff',
        textColor: '#333',
      },
      grid: {
        vertLines: { color: '#f0f0f0' },
        horzLines: { color: '#f0f0f0' },
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // 轉換數據格式 - 使用統一的時間轉換並過濾無效數據
    const candlestickData = validStockData.map(stock => ({
      time: convertTimestamp(stock.timestamp) as Time,
      open: stock.open,
      high: stock.high,
      low: stock.low,
      close: stock.close,
    }))

    console.log('Candlestick data sample:', candlestickData.slice(0, 2))

    // 添加蠟燭圖系列
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderDownColor: '#ef5350',
      borderUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      wickUpColor: '#26a69a',
    })

    candlestickSeries.setData(candlestickData)

    // 添加移動平均線
    if (showMA && maPeriods.length > 0) {
      const colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
      maPeriods.forEach((period, index) => {
        const maKey = `ma_${period}` as keyof StockData
        const maData = validStockData
          .filter(stock => {
            const value = stock[maKey] as number
            return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
          })
          .map(stock => ({
            time: convertTimestamp(stock.timestamp) as Time,
            value: stock[maKey] as number,
          }))

        console.log(`MA${period} data points:`, maData.length)

        if (maData.length > 0) {
          const maSeries = chart.addLineSeries({
            color: colors[index % colors.length],
            lineWidth: 2,
            title: `MA${period}`,
          })
          maSeries.setData(maData)
        }
      })
    }

    // 添加布林帶（放在主圖表）
    if (showBB) {
      // 上軌
      const bbUpperData = validStockData
        .filter(stock => {
          const value = stock.bb_upper
          return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_upper!,
        }))

      console.log('BB Upper data points:', bbUpperData.length)

      if (bbUpperData.length > 0) {
        const bbUpperSeries = chart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'BB Upper',
        })
        bbUpperSeries.setData(bbUpperData)
      }

      // 下軌
      const bbLowerData = validStockData
        .filter(stock => {
          const value = stock.bb_lower
          return value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_lower!,
        }))

      if (bbLowerData.length > 0) {
        const bbLowerSeries = chart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'BB Lower',
        })
        bbLowerSeries.setData(bbLowerData)
      }

      // 中軌
      const bbMiddleData = stockData
        .filter(stock => {
          const value = stock.bb_middle
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.bb_middle!,
        }))

      if (bbMiddleData.length > 0) {
        const bbMiddleSeries = chart.addLineSeries({
          color: '#FFC107',
          lineWidth: 1,
          title: 'BB Middle',
        })
        bbMiddleSeries.setData(bbMiddleData)
      }
    }

    // 添加成交量系列（放在主圖表底部）
    if (showVolume) {
      const volumeData = validStockData.map(stock => ({
        time: convertTimestamp(stock.timestamp) as Time,
        value: stock.volume,
        color: stock.close >= stock.open ? '#26a69a' : '#ef5350',
      }))

      if (volumeData.length > 0) {
        const volumeSeries = chart.addHistogramSeries({
          color: '#26a69a',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        })
        
        volumeSeries.setData(volumeData)

        // 設置成交量的價格比例（放在底部）
        chart.priceScale('volume').applyOptions({
          scaleMargins: {
            top: 0.7,
            bottom: 0,
          },
        })
      }
    }

    // 添加交易信號標記
    if (showSignals && signals.length > 0) {
      const markers = signals.map(signal => ({
        time: convertTimestamp(signal.timestamp) as Time,
        position: (signal.signal_type === 'BUY' ? 'belowBar' : 'aboveBar') as 'belowBar' | 'aboveBar',
        color: signal.signal_type === 'BUY' ? '#26a69a' : '#ef5350',
        shape: (signal.signal_type === 'BUY' ? 'arrowUp' : 'arrowDown') as 'arrowUp' | 'arrowDown',
        text: signal.signal_type === 'BUY' ? '買入' : '賣出',
        size: 2, // 調整箭頭大小 (預設是1，範圍0-4)
      }))
      
      try {
        candlestickSeries.setMarkers(markers)
      } catch (error) {
        console.warn('設置標記時出錯:', error)
      }
    }

    // 自動適應主圖表視圖
    chart.timeScale().fitContent()

    // 儲存圖表實例用於清理
    const charts = [chart]

    // 時間軸同步控制 - 防止無限循環
    let isSyncing = false

    // 時間軸同步 - 當主圖表時間範圍變化時，同步所有子圖表
    const syncTimeRange = (timeRange: any, sourceChart?: any) => {
      if (!timeRange || isSyncing) return // 檢查空值和同步狀態
      
      isSyncing = true
      
      charts.forEach((chartInstance) => {
        if (chartInstance && chartInstance !== sourceChart) {
          try {
            chartInstance.timeScale().setVisibleRange(timeRange)
          } catch (error) {
            console.warn('時間軸同步失敗:', error)
          }
        }
      })
      
      // 延遲重置同步狀態，避免立即觸發
      setTimeout(() => {
        isSyncing = false
      }, 50)
    }

    // 監聽主圖表的時間軸變化
    chart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      syncTimeRange(timeRange, chart)
    })

    // 創建 RSI 子圖表
    let rsiChart: any = null
    if (showRSI) {
      const rsiChartHeight = Math.floor((height - mainChartHeight) / subChartsCount)
      const rsiChartContainer = document.createElement('div')
      rsiChartContainer.style.height = `${rsiChartHeight}px`
      rsiChartContainer.style.marginTop = '10px'
      chartContainerRef.current.appendChild(rsiChartContainer)

      rsiChart = createChart(rsiChartContainer, {
        width: chartContainerRef.current.clientWidth,
        height: rsiChartHeight,
        layout: {
          backgroundColor: '#ffffff',
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#f0f0f0' },
          horzLines: { color: '#f0f0f0' },
        },
        rightPriceScale: {
          borderColor: '#cccccc',
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: '#cccccc',
          timeVisible: false,
          secondsVisible: false,
        },
      })

      const rsiData = stockData
        .filter(stock => {
          const value = stock.rsi
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.rsi!,
        }))

      if (rsiData.length > 0) {
        const rsiSeries = rsiChart.addLineSeries({
          color: '#9C27B0',
          lineWidth: 2,
          title: 'RSI',
        })
        rsiSeries.setData(rsiData)

        // 添加 RSI 的 30 和 70 參考線
        const rsiRef30 = rsiChart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'RSI 30',
        })
        const rsiRef70 = rsiChart.addLineSeries({
          color: '#FF5722',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'RSI 70',
        })

        const ref30Data = rsiData.map(item => ({ time: item.time, value: 30 }))
        const ref70Data = rsiData.map(item => ({ time: item.time, value: 70 }))
        
        rsiRef30.setData(ref30Data)
        rsiRef70.setData(ref70Data)

        // 設置 RSI 圖表的價格範圍
        rsiChart.priceScale().applyOptions({
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        })
      }

      rsiChart.timeScale().fitContent()
      charts.push(rsiChart)

      // 為 RSI 圖表添加反向時間軸同步
      rsiChart.timeScale().subscribeVisibleTimeRangeChange((timeRange: any) => {
        syncTimeRange(timeRange, rsiChart)
      })
    }

    // 創建 MACD 子圖表
    let macdChart: any = null
    if (showMACD) {
      const macdChartHeight = Math.floor((height - mainChartHeight) / subChartsCount)
      const macdChartContainer = document.createElement('div')
      macdChartContainer.style.height = `${macdChartHeight}px`
      macdChartContainer.style.marginTop = '10px'
      chartContainerRef.current.appendChild(macdChartContainer)

      macdChart = createChart(macdChartContainer, {
        width: chartContainerRef.current.clientWidth,
        height: macdChartHeight,
        layout: {
          backgroundColor: '#ffffff',
          textColor: '#333',
        },
        grid: {
          vertLines: { color: '#f0f0f0' },
          horzLines: { color: '#f0f0f0' },
        },
        rightPriceScale: {
          borderColor: '#cccccc',
          scaleMargins: {
            top: 0.1,
            bottom: 0.1,
          },
        },
        timeScale: {
          borderColor: '#cccccc',
          timeVisible: true,
          secondsVisible: false,
        },
      })

      const macdData = stockData
        .filter(stock => {
          const value = stock.macd
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd!,
        }))

      const macdSignalData = stockData
        .filter(stock => {
          const value = stock.macd_signal
          return stock && value !== null && value !== undefined && !isNaN(value) && isFinite(value)
        })
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd_signal!,
        }))

      const macdHistogramData = stockData
        .filter(stock => 
          stock && 
          stock.macd !== null && stock.macd !== undefined && 
          stock.macd_signal !== null && stock.macd_signal !== undefined &&
          !isNaN(stock.macd) && !isNaN(stock.macd_signal) &&
          isFinite(stock.macd) && isFinite(stock.macd_signal)
        )
        .map(stock => ({
          time: convertTimestamp(stock.timestamp) as Time,
          value: stock.macd! - stock.macd_signal!,
          color: (stock.macd! - stock.macd_signal!) >= 0 ? '#26a69a' : '#ef5350',
        }))

      if (macdData.length > 0) {
        // MACD 線
        const macdSeries = macdChart.addLineSeries({
          color: '#2196F3',
          lineWidth: 2,
          title: 'MACD',
        })
        macdSeries.setData(macdData)

        // Signal 線
        if (macdSignalData.length > 0) {
          const macdSignalSeries = macdChart.addLineSeries({
            color: '#FF9800',
            lineWidth: 2,
            title: 'Signal',
          })
          macdSignalSeries.setData(macdSignalData)
        }

        // MACD 柱狀圖
        if (macdHistogramData.length > 0) {
          const macdHistogramSeries = macdChart.addHistogramSeries({
            color: '#26a69a',
            priceFormat: {
              type: 'price',
              precision: 4,
              minMove: 0.0001,
            },
          })
          macdHistogramSeries.setData(macdHistogramData)
        }

        // 添加零軸參考線
        const zeroLineData = macdData.map(item => ({ time: item.time, value: 0 }))
        const zeroLineSeries = macdChart.addLineSeries({
          color: '#666666',
          lineWidth: 1,
          lineStyle: 2, // dashed
          title: 'Zero Line',
        })
        zeroLineSeries.setData(zeroLineData)
      }

      macdChart.timeScale().fitContent()
      charts.push(macdChart)

      // 為 MACD 圖表添加反向時間軸同步
      macdChart.timeScale().subscribeVisibleTimeRangeChange((timeRange: any) => {
        syncTimeRange(timeRange, macdChart)
      })
    }

    // 響應式調整 - 為所有圖表設置
    const handleResize = () => {
      if (chartContainerRef.current) {
        charts.forEach(chartInstance => {
          if (chartInstance) {
            chartInstance.applyOptions({
              width: chartContainerRef.current!.clientWidth,
            })
          }
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      // 清理所有圖表實例
      charts.forEach(chartInstance => {
        if (chartInstance) {
          chartInstance.remove()
        }
      })
    }
  }, [stockData, signals, showSignals, height, showMA, maPeriods, showRSI, showBB, showMACD, showVolume])

  return (
    <div className="w-full">
      <div 
        ref={chartContainerRef} 
        className="w-full border rounded-lg"
        style={{ height: `${height}px` }}
      />
      
      {/* 圖例 */}
      <div className="flex flex-wrap justify-center mt-4 space-x-4 text-sm">
        {/* 主圖指標 */}
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1">
            <div className="w-2 h-4 bg-green-600"></div>
            <div className="w-2 h-4 bg-red-500"></div>
          </div>
          <span>K線圖</span>
        </div>
        {showVolume && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gray-400 rounded"></div>
            <span>成交量</span>
          </div>
        )}
        {showMA && maPeriods.map((period, index) => {
          const colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
          return (
            <div key={`ma-${index}-${period}`} className="flex items-center space-x-2">
              <div 
                className="w-3 h-1"
                style={{ backgroundColor: colors[index % colors.length] }}
              ></div>
              <span>MA{period}</span>
            </div>
          )
        })}
        {showBB && (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-1 bg-red-600"></div>
              <div className="w-2 h-1 bg-yellow-500"></div>
            </div>
            <span>布林帶</span>
          </div>
        )}
        
        {/* 子圖指標 */}
        {showRSI && (
          <div className="flex items-center space-x-2">
            <div className="w-3 h-1 bg-purple-600"></div>
            <span>RSI (子圖)</span>
          </div>
        )}
        {showMACD && (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-1 bg-blue-500"></div>
              <div className="w-2 h-1 bg-orange-500"></div>
              <div className="w-2 h-2 bg-green-600"></div>
            </div>
            <span>MACD (子圖)</span>
          </div>
        )}
        
        {showSignals && (
          <>
            <div className="flex items-center space-x-2">
              <span className="text-green-600 text-lg font-bold">▲</span>
              <span className="font-medium">買入信號</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-red-500 text-lg font-bold">▼</span>
              <span className="font-medium">賣出信號</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}