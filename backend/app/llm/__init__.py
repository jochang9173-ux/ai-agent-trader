"""
LLM modules package initialization
"""

from . import analysis
from .analysis import EnhancedTrendAnalyzer, EnhancedTrendResult, TrendAnalysisResult
from .client import LLMClientConfig, get_configured_client, get_llm_client
from .strategies import (
    # LLMSmartStrategy,  # Temporarily commented out due to refactored module removal
    ParameterSpec,
    ParameterType,
    SignalType,
    StrategyConfig,
    TradingSignal,
    TradingStrategy,
    get_available_strategies,
)

__all__ = [
    # Client utilities
    "get_llm_client",
    "LLMClientConfig",
    "get_configured_client",
    # Analysis Tools
    "EnhancedTrendAnalyzer",
    "EnhancedTrendResult",
    "TrendAnalysisResult",
    # Strategy Framework
    "ParameterSpec",
    "ParameterType",
    "SignalType",
    "StrategyConfig",
    "TradingSignal",
    "TradingStrategy",
    "get_available_strategies",
    # LLM Strategy (Temporarily commented out due to refactored module removal)
    # "LLMSmartStrategy",
    # Sub-modules
    "analysis",
]
