"""
LLM Strategies Module
基於 LLM 的交易策略 - 領域分離架構

此模塊實現了重構的LLM交易策略，使用領域分離架構將原有的單體類分解為多個專業模塊：

核心模塊:
- LLMSmartStrategy: 主策略類 (重構版)
- LLMDecisionEngine: LLM決策引擎
- RiskManager: 風險管理器
- PositionManager: 倉位管理器
- StockCharacteristicsAnalyzer: 股票特性分析器
- TradingEventDetector: 交易事件檢測器
- PerformanceTracker: 績效追蹤器

重構前後對比:
- 重構前: 單一類 (~2900行, 30+方法)
- 重構後: 8個專業模塊 (~500行/模塊, 職責明確)
"""

from .base import (
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
    get_available_strategies,
)

# 新的重構模塊 (暫時註解掉，因為data_types已移除)
# from .data_types import (
#     # 決策相關
#     DecisionContext,
#     LLMDecision,
#     PerformanceMetrics,
#     PnLInsights,
#     PositionMetrics,
#     StockCharacteristics,
#     # 策略狀態
#     StrategyState,
#     TechnicalParameters,
#     # 核心數據類型
#     TradingEvent,
#     TradingSignalRequest,
# )
# from .llm_decision_engine import LLMDecisionEngine
# from .llm_smart_strategy import LLMSmartStrategy

# 原有策略 (向後兼容)
from .llm_strategy import LLMSmartStrategy as LLMStrategyLegacy
# from .performance_tracker import PerformanceTracker
# from .position_manager import PositionManager
# from .risk_manager import RiskManager
# from .stock_characteristics_analyzer import StockCharacteristicsAnalyzer
# from .trading_event_detector import TradingEventDetector

__all__ = [
    # Base Classes
    "ParameterSpec",
    "ParameterType", 
    "SignalType",
    "StrategyConfig",
    "TradingSignal",
    "TradingStrategy",
    "get_available_strategies",
    # 原有策略 (向後兼容)
    "LLMStrategyLegacy",
    # 新的主策略類 (暫時註解，因為模塊已移除)
    # "LLMSmartStrategy",
    # 核心模塊 (暫時註解，因為模塊已移除)
    # "LLMDecisionEngine",
    # "RiskManager", 
    # "PositionManager",
    # "StockCharacteristicsAnalyzer",
    # "TradingEventDetector",
    # "PerformanceTracker",
    # 數據類型 (暫時註解，因為模塊已移除)
    # "TradingEvent",
    # "StockCharacteristics",
    # "TechnicalParameters",
    # "PositionMetrics",
    # "PnLInsights",
    # "PerformanceMetrics", 
    # "DecisionContext",
    # "LLMDecision",
    # "TradingSignalRequest",
    # "StrategyState",
    # 輔助函數
    "print_architecture_info",
    "get_module_info",
]

# 版本信息
__version__ = "2.0.0"
__author__ = "LLM Agent Trader Team"
__description__ = "Refactored LLM Trading Strategy with Domain Separation Architecture"

# 架構說明
ARCHITECTURE_INFO = """
領域分離架構 (Domain Separation Architecture):

📊 LLMSmartStrategy (主控制器)
├── 🤖 LLMDecisionEngine (LLM決策引擎)
│   ├── prompt建構
│   ├── LLM調用
│   └── 響應解析
├── ⚡ RiskManager (風險管理器)  
│   ├── 風險評估
│   ├── 損益洞察
│   └── 決策驗證
├── 💼 PositionManager (倉位管理器)
│   ├── 持倉追蹤
│   ├── 交易執行
│   └── 損益計算
├── 📈 StockCharacteristicsAnalyzer (股票特性分析器)
│   ├── 波動性分析
│   ├── 趨勢一致性
│   └── MACD有效性
├── 🔍 TradingEventDetector (交易事件檢測器)
│   ├── MACD信號
│   ├── 均線穿越
│   ├── 布林帶突破
│   └── 價格突破
└── 📊 PerformanceTracker (績效追蹤器)
    ├── 交易記錄
    ├── 績效計算
    └── 報告生成

📋 data_types (共享數據結構)
├── DTOs和數據類
├── 類型定義
└── 接口標準
"""


def print_architecture_info():
    """打印架構信息"""
    print(ARCHITECTURE_INFO)


def get_module_info():
    """獲取模塊信息"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "modules": len(__all__),
        "architecture": "Domain Separation Architecture",
    }
