# LLM Agent Trader 系統架構設計

## 系統概述
LLM Agent Trader 是一個基於人工智慧的股票交易回測系統，整合了大語言模型(LLM)進行智能交易決策分析。

## 高層次系統流程圖

```mermaid
flowchart TD
    %% 用戶界面層
    A[用戶界面 - Next.js Frontend] --> B[API Gateway - FastAPI Backend]
    
    %% 主要功能模組
    B --> C[LLM 流式回測引擎]
    B --> D[回測分析 API]
    B --> E[每日反饋 API]
    
    %% 數據層
    F[股票數據服務<br/>YFinance] --> C
    G[SQLite 數據庫<br/>回測日誌] --> D
    G --> E
    
    %% LLM 策略引擎
    C --> H[LLM 智能策略]
    H --> I[Azure OpenAI<br/>GPT-4]
    H --> J[技術分析引擎]
    H --> K[風險管理模組]
    
    %% 回測執行流程
    C --> L[交易信號生成]
    L --> M[績效計算]
    M --> N[結果記錄]
    N --> G
    
    %% 樣式定義
    classDef frontend fill:#e1f5fe
    classDef backend fill:#f3e5f5
    classDef llm fill:#fff3e0
    classDef data fill:#e8f5e8
    classDef tools fill:#fce4ec
    
    class A frontend
    class B,C,D,E backend
    class H,I,J,K llm
    class F,G data
```

## 核心概念說明

### LLM 流式回測引擎 vs 回測引擎
- **LLM 流式回測引擎**: 是整個回測系統的協調層，負責：
  - 接收前端請求並初始化回測流程
  - 使用 Server-Sent Events 提供即時進度更新
  - 協調數據獲取、策略執行、結果處理等各個環節
  - 管理回測會話和狀態

- **回測引擎**: 是核心計算引擎，專注於：
  - 執行具體的回測邏輯和交易模擬
  - 處理 LLM 策略產生的交易信號
  - 計算績效指標和風險指標
  - 維護持倉狀態和資金管理

簡單來說，**流式回測引擎**是整個系統的"指揮官"，而**回測引擎**是執行具體計算的"執行者"。

## 詳細系統架構

### 1. 前端層 (Frontend)
```mermaid
flowchart LR
    A[Next.js 應用] --> B[React 組件]
    B --> C[圖表組件<br/>Recharts/LightweightCharts]
    B --> D[分析面板]
    B --> E[LLM 對話界面]
    
    F[狀態管理<br/>React Query] --> B
    G[UI 組件庫<br/>Radix UI] --> B
```

### 2. 後端 API 層
```mermaid
flowchart TD
    A[FastAPI 主應用] --> B[API 路由 v1]
    
    B --> C[LLM 流式回測<br/>/llm-stream]
    B --> D[回測分析<br/>/backtest]
    B --> E[每日反饋<br/>/daily]
    
    C --> F[Server-Sent Events<br/>即時進度更新]
    D --> G[歷史數據查詢<br/>回顧分析]
    E --> H[決策改善建議<br/>策略優化]
```

### 3. LLM 策略引擎
```mermaid
flowchart TD
    A[LLM 智能策略] --> B[市場數據分析]
    A --> C[技術指標計算]
    A --> D[趨勢識別]
    
    B --> E[Azure OpenAI API]
    C --> F[移動平均線<br/>MACD, RSI, 布林帶]
    D --> G[短期/中期/長期趨勢]
    
    E --> H[決策推理生成]
    F --> I[技術事件觸發]
    G --> J[市場狀態評估]
    
    H --> K[交易決策輸出]
    I --> K
    J --> K
    
    K --> L[BUY/SELL/HOLD]
    K --> M[信心度評分]
    K --> N[風險評估]
```

### 4. 數據流程
```mermaid
flowchart LR
    A[Yahoo Finance API] --> B[股票數據獲取]
    B --> C[數據預處理<br/>OHLCV格式]
    C --> D[技術指標計算]
    D --> E[LLM 分析處理]
    E --> F[決策記錄]
    F --> G[SQLite 數據庫]
    
    G --> H[回測分析查詢]
    G --> I[績效統計]
    G --> J[歷史回顧]
```

### 5. 回測執行流程
```mermaid
flowchart TD
    A[開始回測] --> B[載入股票數據]
    B --> C[初始化策略參數]
    C --> D[逐日數據處理]
    
    D --> E[技術分析計算]
    E --> F[LLM 決策推理]
    F --> G[交易信號生成]
    
    G --> H{是否有交易信號?}
    H -->|是| I[執行交易]
    H -->|否| J[保持現狀]
    
    I --> K[更新持倉狀態]
    J --> K
    K --> L[記錄日誌]
    L --> M[計算績效指標]
    
    M --> N{還有數據?}
    N -->|是| D
    N -->|否| O[生成最終報告]
    
    O --> P[返回結果]
```

## 核心功能模組

### 1. LLM 決策引擎
- **輸入**: 市場數據、技術指標、歷史趨勢
- **處理**: Azure OpenAI GPT-4 推理分析
- **輸出**: 交易決策、信心度、推理過程

### 2. 技術分析模組
- **移動平均線**: SMA, EMA 交叉策略
- **動量指標**: RSI, MACD 信號
- **波動性指標**: 布林帶突破
- **趨勢識別**: 多時間框架分析

### 3. 風險管理
- **停損機制**: 固定比例停損
- **停利設定**: 目標利潤鎖定
- **倉位控制**: 最大持倉比例限制
- **資金管理**: 動態資金分配

### 4. 數據持久化
- **SQLite 數據庫**: 輕量級本地存儲
- **日誌記錄**: 每日交易決策和分析
- **績效追蹤**: 累積收益和風險指標
- **事件記錄**: 技術事件和觸發條件

## 技術棧

### 前端技術
- **框架**: Next.js 15.4.4 (React 19)
- **樣式**: Tailwind CSS 4.0
- **圖表**: Recharts, Lightweight Charts
- **狀態管理**: TanStack React Query
- **UI 組件**: Radix UI

### 後端技術
- **API 框架**: FastAPI (Python)
- **數據庫**: SQLite
- **LLM 集成**: Azure OpenAI API
- **數據源**: Yahoo Finance (yfinance)

### 開發和運維
- **開發環境**: Python 虛擬環境 + Node.js
- **版本控制**: Git
- **依賴管理**: uv (Python), npm (Node.js)

## 系統特色

1. **即時流式回測**: 使用 Server-Sent Events 提供即時進度更新
2. **智能決策分析**: 整合 GPT-4 進行深度市場分析
3. **互動式反饋**: 支持用戶對歷史決策提供反饋和改善建議
4. **全面技術分析**: 多維度技術指標和趨勢識別
5. **歷史數據分析**: 支持複雜的回測數據查詢和回顧分析

## 擴展性設計

- **模組化架構**: 各功能模組獨立，易於擴展
- **API 版本控制**: 支持多版本 API 並存
- **策略插件化**: 支持多種交易策略並行
- **多數據源**: 可擴展支持更多金融數據提供商
- **部署靈活性**: 支持本地開發和雲端部署