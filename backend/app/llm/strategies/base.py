"""
交易策略基础框架
提供标准化的策略接口，專為 LLM 策略設計
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SignalType(Enum):
    """交易信号类型"""

    BUY = 1  # 買入信號（開多倉）
    SELL = -1  # 賣出信號（平多倉或開空倉，根據trading_mode決定）
    HOLD = 0  # 持有信號

    # 為LLM多空選擇的明確信號類型
    LONG_OPEN = 11  # 明確開多倉
    LONG_CLOSE = -11  # 明確平多倉
    SHORT_OPEN = 12  # 明確開空倉
    SHORT_CLOSE = -12  # 明確平空倉


class ParameterType(Enum):
    """參數類型枚舉"""

    INTEGER = "integer"  # 整數
    FLOAT = "float"  # 浮點數
    BOOLEAN = "boolean"  # 布林值
    SELECT = "select"  # 選擇項
    RANGE = "range"  # 範圍值


@dataclass
class ParameterSpec:
    """
    參數規格定義
    用於前端動態生成參數調整界面
    """

    name: str  # 參數名稱
    display_name: str  # 顯示名稱
    description: str  # 參數說明
    param_type: ParameterType  # 參數類型
    default_value: Any = None  # 預設值
    min_value: Optional[float] = None  # 最小值（數值類型）
    max_value: Optional[float] = None  # 最大值（數值類型）
    step: Optional[float] = None  # 步長（數值類型）
    options: Optional[List[Any]] = None  # 選項列表（選擇類型）
    required: bool = True  # 是否必填


@dataclass
class StrategyConfig:
    """策略配置"""

    name: str  # 策略名稱
    description: str  # 策略描述
    parameters: Dict[str, Any] = field(default_factory=dict)  # 策略參數


@dataclass
class TradingSignal:
    """
    交易信號
    包含具體的交易執行信息
    """

    signal_type: SignalType  # 信號類型
    timestamp: pd.Timestamp  # 信號時間
    price: float  # 建議價格
    quantity: int = 0  # 交易數量（0表示按比例）
    confidence: float = 0.0  # 信心度 0-1
    reason: str = ""  # 交易原因
    stop_loss: Optional[float] = None  # 止損價格
    take_profit: Optional[float] = None  # 止盈價格
    metadata: Dict[str, Any] = field(default_factory=dict)  # 額外信息

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "signal_type": self.signal_type.name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "price": self.price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "reason": self.reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


class TradingStrategy(ABC):
    """
    交易策略抽象基類
    專為 LLM 策略設計的簡化版本
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.description = config.description
        self.parameters = config.parameters

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        生成交易信號

        Args:
            data: 市場數據 DataFrame

        Returns:
            交易信號列表
        """
        pass

    def get_parameter_specs(self) -> List[ParameterSpec]:
        """
        獲取策略參數規格
        子類應重寫此方法以提供參數定義
        """
        return []

    def update_parameters(self, parameters: Dict[str, Any]):
        """更新策略參數"""
        self.parameters.update(parameters)

    def get_info(self) -> Dict[str, Any]:
        """獲取策略基本信息"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "parameter_specs": [spec.__dict__ for spec in self.get_parameter_specs()],
        }


def get_available_strategies() -> Dict[str, Dict[str, Any]]:
    """
    獲取可用的策略列表
    現在只包含 LLM 策略 (使用原版)
    """
    from .llm_strategy import LLMSmartStrategy  # 切換回原版

    strategies = {
        "llm_smart": {
            "name": "LLM Smart Strategy",
            "description": "基於大語言模型的智能交易策略",
            "class": LLMSmartStrategy,
            "category": "AI/LLM",
            "parameters": [
                {
                    "name": "confidence_threshold",
                    "display_name": "信心度閾值",
                    "description": "執行交易的最低信心度要求",
                    "type": "float",
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                },
                {
                    "name": "max_daily_trades",
                    "display_name": "每日最大交易次數",
                    "description": "限制每日交易頻率",
                    "type": "integer",
                    "default": 3,
                    "min": 1,
                    "max": 10,
                },
                {
                    "name": "max_loss_threshold",
                    "display_name": "最大損失閾值",
                    "description": "觸發止損的最大損失比例",
                    "type": "float",
                    "default": 0.05,
                    "min": 0.01,
                    "max": 0.2,
                    "step": 0.01,
                },
            ],
        }
    }

    return strategies
