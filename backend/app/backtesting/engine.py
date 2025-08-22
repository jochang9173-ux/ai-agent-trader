"""
自建回測引擎模組

自建的回測引擎，提供透明、易懂的交易邏輯
符合一般投資人的交易習慣和理解方式
設計為LLM友好的工具，提供簡潔的API介面
"""

import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..llm.strategies import SignalType, TradingSignal, TradingStrategy
from ..utils.fetcher import StockDataFetcher
from ..utils.indicators import (
    calculate_bollinger_bands,
    calculate_macd,
    calculate_moving_averages,
    calculate_rsi,
)

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """交易模式列舉"""

    LONG_ONLY = "long_only"  # 只做多
    SHORT_ONLY = "short_only"  # 只做空
    LONG_SHORT = "long_short"  # 多空自由


class OrderType(Enum):
    """訂單類型列舉"""

    BUY = "buy"
    SELL = "sell"


class TradeStatus(Enum):
    """交易狀態列舉"""

    PENDING = "pending"  # 待執行
    EXECUTED = "executed"  # 已執行
    CANCELLED = "cancelled"  # 已取消
    FAILED = "failed"  # 執行失敗


@dataclass
class Trade:
    """
    交易記錄 - 記錄單筆交易的完整資訊
    """

    trade_id: str  # 交易編號
    timestamp: datetime  # 交易時間
    symbol: str  # 股票代碼
    order_type: OrderType  # 買入或賣出
    shares: int  # 交易股數
    price: float  # 交易價格
    commission: float  # 手續費
    total_cost: float  # 總成本（含手續費）
    status: TradeStatus  # 交易狀態
    signal_confidence: float = 0.0  # 信號強度
    reason: str = ""  # 交易原因

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，確保所有值都是 JSON 可序列化的"""
        return {
            "trade_id": str(self.trade_id),
            "timestamp": self.timestamp.isoformat()
            if hasattr(self.timestamp, "isoformat")
            else str(self.timestamp),
            "symbol": str(self.symbol),
            "order_type": self.order_type.value,
            "shares": int(self.shares),
            "price": float(self.price),
            "total_cost": float(self.total_cost),
            "commission": float(self.commission)
            if self.commission is not None
            else 0.0,
            "status": self.status.value,
            "signal_confidence": float(self.signal_confidence),
            "reason": str(self.reason) if self.reason else "",
        }


@dataclass
class Portfolio:
    """
    投資組合狀態 - 追蹤現金和持股狀況
    """

    cash: float = 0.0  # 現金餘額
    positions: Dict[str, int] = None  # 持股數量 {股票代碼: 股數}

    def __post_init__(self):
        if self.positions is None:
            self.positions = {}

    def get_position(self, symbol: str) -> int:
        """取得特定股票的持股數量"""
        return self.positions.get(symbol, 0)

    def update_position(self, symbol: str, shares: int) -> None:
        """更新持股數量"""
        if shares == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = shares

    def calculate_total_value(self, prices: Dict[str, float]) -> float:
        """計算投資組合總價值（現金 + 持股市值）"""
        stock_value = sum(
            shares * prices.get(symbol, 0.0)
            for symbol, shares in self.positions.items()
        )
        return self.cash + stock_value

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，適合LLM處理"""
        return {"cash": self.cash, "positions": self.positions.copy()}


@dataclass
class BacktestConfig:
    """
    回測設定類別 - 簡化且LLM友好的設計
    移除了保證金倍數和最大倉位比例的概念
    改為一般投資人容易理解的最大持股數量
    """

    initial_capital: float = 1000000.0  # 初始資金
    max_shares_per_trade: int = 1000  # 每次最大買入股數
    trading_mode: TradingMode = TradingMode.LONG_ONLY  # 交易模式
    trade_on_open: bool = False  # 是否在開盤價交易（False=收盤價）
    commission_rate: float = 0.001425  # 手續費率（台股約0.1425%）
    min_commission: float = 20.0  # 最低手續費（台股20元）

    def calculate_commission(self, trade_value: float) -> float:
        """計算手續費"""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，方便LLM處理"""
        return {
            "initial_capital": self.initial_capital,
            "max_shares_per_trade": self.max_shares_per_trade,
            "trading_mode": self.trading_mode.value,
            "trade_on_open": self.trade_on_open,
            "commission_rate": self.commission_rate,
            "min_commission": self.min_commission,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestConfig":
        """從字典建立設定，方便LLM呼叫"""
        config_data = data.copy()
        if "trading_mode" in config_data:
            config_data["trading_mode"] = TradingMode(config_data["trading_mode"])
        return cls(**config_data)


class CustomBacktestEngine:
    """
    自建回測引擎 - 簡潔、透明的交易邏輯

    專為一般投資人設計，避免複雜的保證金和倉位概念
    提供清晰的現金流管理和持股追蹤
    適合LLM工具調用的簡潔API
    """

    def __init__(self, config: BacktestConfig = None):
        """
        初始化回測引擎

        Args:
            config: 回測設定，為None時使用預設設定
        """
        self.config = config or BacktestConfig()
        self.strategies: Dict[str, TradingStrategy] = {}
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # 回測狀態
        self.is_running = False
        self.current_results: Dict[str, Any] = {}

    def add_strategy(self, name: str, strategy: TradingStrategy) -> None:
        """
        新增策略

        Args:
            name: 策略名稱
            strategy: 策略物件
        """
        self.strategies[name] = strategy
        logger.info(f"新增策略: {name}")

    def load_data(
        self,
        symbol: str,
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        載入回測數據

        Args:
            symbol: 股票代碼
            period: 數據週期（1y, 6mo, 3mo等）
            start_date: 開始日期 (可選)
            end_date: 結束日期 (可選)

        Returns:
            處理後的股票數據
        """
        logger.info(f"載入 {symbol} 數據，週期: {period}")

        try:
            # 獲取數據
            data = StockDataFetcher.fetch_stock_data(symbol, period)

            if data is None or data.empty:
                raise ValueError(f"無法取得 {symbol} 的數據")

            # 標準化欄位名稱
            data.columns = data.columns.str.lower()

            # 根據分析期間篩選數據
            if hasattr(data, "attrs") and "analysis_start_date" in data.attrs:
                analysis_start = pd.to_datetime(data.attrs["analysis_start_date"])

                # 處理時區相容性問題
                if data.index.tz is not None and analysis_start.tz is None:
                    analysis_start = analysis_start.tz_localize(data.index.tz)
                elif data.index.tz is None and analysis_start.tz is not None:
                    analysis_start = analysis_start.tz_localize(None)

                logger.info(
                    f"篩選數據至分析期間: 從 {analysis_start.date()} 到 {data.index.max().date()}"
                )
                data = data[analysis_start:]

            # 如果另外指定了日期範圍，進行額外篩選
            if start_date and end_date:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                data = data[start:end]

            # 確保數據類型正確
            for col in ["open", "high", "low", "close", "volume"]:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")

            # 移除空值
            data = data.dropna()

            # 快取數據
            self.data_cache[symbol] = data

            logger.info(f"載入 {len(data)} 筆 {symbol} 數據")
            logger.info(f"日期範圍: {data.index[0]} 到 {data.index[-1]}")

            return data

        except Exception as e:
            logger.error(f"載入 {symbol} 數據時發生錯誤: {e}")
            raise

    def run_backtest(
        self,
        stock_data: pd.DataFrame,
        strategy: TradingStrategy,
        initial_cash: float = 10000.0,
        transaction_cost: float = 0.001,
        symbol: str = None,
    ) -> Dict[str, Any]:
        """
        執行回測 - 自建引擎的核心方法

        Args:
            stock_data: 股票數據DataFrame
            strategy: 策略物件 (可以是單一策略或組合策略)
            initial_cash: 初始資金
            transaction_cost: 交易成本比例
            symbol: 股票代碼

        Returns:
            回測結果字典
        """
        self.is_running = True

        try:
            # 確定策略名稱
            strategy_name = getattr(strategy, "name", type(strategy).__name__)

            logger.info(f"開始回測，使用策略 {strategy_name}")

            # 1. 初始化投資組合狀態
            portfolio = Portfolio(cash=initial_cash)

            # 2. 計算技術指標
            enhanced_data = self._prepare_technical_indicators(stock_data.copy())

            # 3. 生成交易信號
            # 設置當前股票代碼到策略中，供 Enhanced 趨勢分析使用
            if hasattr(strategy, "set_current_symbol"):
                strategy.set_current_symbol(symbol or "UNKNOWN")
            elif hasattr(strategy, "current_symbol"):
                strategy.current_symbol = symbol or "UNKNOWN"

            # 設置策略的初始資金，確保與 portfolio 一致
            if hasattr(strategy, "cash") and hasattr(strategy, "initial_capital"):
                strategy.initial_capital = initial_cash
                strategy.cash = initial_cash
                strategy.current_portfolio_value = initial_cash
                strategy.max_portfolio_value = initial_cash
                print(f"💰 設置策略初始資金: ${initial_cash:,.0f}")

            signals = strategy.generate_signals(enhanced_data)

            logger.info(f"生成了 {len(signals)} 個交易信號")

            # 4. 執行交易模擬
            trades, portfolio_history = self._simulate_trading(
                signals,
                enhanced_data,
                portfolio,
                symbol=symbol or "UNKNOWN",
                transaction_cost=transaction_cost,
                initial_cash=initial_cash,
            )

            # 4.5. 回測結束處理 - 強制結算持倉
            final_date = enhanced_data.index[-1]
            final_price = float(enhanced_data.iloc[-1]["close"])

            # 如果策略有 finalize_backtest 方法，則調用它
            if hasattr(strategy, "finalize_backtest"):
                try:
                    strategy.finalize_backtest(final_price, final_date)
                    logger.info(f"策略 {strategy_name} 已執行回測結束處理")
                except Exception as e:
                    logger.warning(f"策略 {strategy_name} 回測結束處理失敗: {e}")

            # 如果投資組合還有持倉，強制結算
            current_position = portfolio.get_position(symbol or "UNKNOWN")
            if current_position > 0:
                logger.info(f"檢測到未結算持倉 {current_position} 股，強制結算")

                # 創建強制結算交易
                final_trade = Trade(
                    trade_id=f"FINAL_{len(trades)}",
                    timestamp=final_date,
                    symbol=symbol or "UNKNOWN",
                    order_type=OrderType.SELL,
                    shares=current_position,
                    price=final_price,
                    commission=0.0,  # 回測結束不收手續費
                    total_cost=current_position * final_price,
                    status=TradeStatus.EXECUTED,
                    signal_confidence=1.0,
                    reason="回測結束強制結算",
                )

                # 執行強制結算
                portfolio.cash += current_position * final_price
                portfolio.update_position(symbol or "UNKNOWN", 0)  # 清空持倉
                trades.append(final_trade)

                # 更新投資組合歷史的最後一筆記錄
                if portfolio_history:
                    portfolio_history[-1].update(
                        {
                            "cash": portfolio.cash,
                            "position": 0,
                            "stock_value": 0,
                            "total_value": portfolio.cash,
                            "cumulative_return": (portfolio.cash - initial_cash)
                            / initial_cash,
                        }
                    )

                logger.info(
                    f"強制結算完成: 售出 {current_position} 股，價格 ${final_price:.2f}"
                )

            # 5. 計算績效指標
            results = self._calculate_performance_metrics(
                portfolio_history,
                enhanced_data,
                trades,
                signals,
                symbol=symbol or "UNKNOWN",
                strategy_name=strategy_name,
                strategy=strategy,  # 傳遞策略對象
                initial_cash=initial_cash,
            )

            # 6. 儲存結果
            self.current_results[
                f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ] = results

            logger.info(f"回測完成: {strategy_name}")
            return results

        except Exception as e:
            logger.error(f"回測失敗: {e}")
            raise
        finally:
            self.is_running = False

    def _prepare_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        準備技術指標數據

        Args:
            data: 原始股價數據

        Returns:
            包含技術指標的數據
        """
        # 計算各種技術指標
        if "5ma" not in data.columns:
            data = calculate_moving_averages(data)
        if "bb_upper" not in data.columns:
            data = calculate_bollinger_bands(data)
        if "rsi" not in data.columns:
            data = calculate_rsi(data)
        if "macd" not in data.columns:
            data = calculate_macd(data)

        return data

    def _simulate_trading(
        self,
        signals: List[TradingSignal],
        data: pd.DataFrame,
        portfolio: Portfolio,
        symbol: str,
        transaction_cost: float = 0.001,
        initial_cash: float = 10000.0,
    ) -> Tuple[List[Trade], List[Dict[str, Any]]]:
        """
        模擬交易執行 - 核心交易邏輯

        Args:
            signals: 交易信號列表
            data: 股價數據
            portfolio: 投資組合狀態
            symbol: 股票代碼

        Returns:
            (交易記錄列表, 投資組合歷史)
        """
        trades = []
        portfolio_history = []
        trade_counter = 0

        # 建立信號查找表
        signal_dict = {}
        for signal in signals:
            date_key = signal.timestamp.date()
            if date_key not in signal_dict:
                signal_dict[date_key] = []
            signal_dict[date_key].append(signal)

        # 逐日模擬交易
        for date, row in data.iterrows():
            current_date = date.date() if hasattr(date, "date") else date
            current_price = float(row["close"])

            # 檢查當日是否有信號
            daily_signals = signal_dict.get(current_date, [])

            # 處理交易信號
            for signal in daily_signals:
                if signal.signal_type == SignalType.BUY:
                    trade = self._execute_buy_order(
                        signal,
                        current_price,
                        portfolio,
                        symbol,
                        trade_counter,
                        transaction_cost,
                    )
                    if trade:
                        trades.append(trade)
                        trade_counter += 1

                elif signal.signal_type == SignalType.SELL:
                    trade = self._execute_sell_order(
                        signal,
                        current_price,
                        portfolio,
                        symbol,
                        trade_counter,
                        transaction_cost,
                    )
                    if trade:
                        trades.append(trade)
                        trade_counter += 1

            # 記錄當日投資組合狀態
            current_position = portfolio.get_position(symbol)
            stock_value = current_position * current_price
            total_value = portfolio.cash + stock_value

            # 計算累積報酬率（相對於初始資金）
            cumulative_return = (total_value - initial_cash) / initial_cash

            # 計算未實現損益和本次交易收益率
            unrealized_pnl = 0.0
            unrealized_pnl_pct = 0.0
            position_entry_price = 0.0

            # 如果有持倉，計算未實現損益
            if current_position > 0:
                # 從最近的買入交易中找到進場價格
                recent_buy_trades = [t for t in trades if t.order_type.value == "buy"]
                if recent_buy_trades:
                    latest_buy_trade = recent_buy_trades[-1]
                    position_entry_price = latest_buy_trade.price
                    cost_basis = current_position * position_entry_price
                    unrealized_pnl = stock_value - cost_basis
                    unrealized_pnl_pct = (
                        (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
                    )

            portfolio_snapshot = {
                "date": date,
                "cash": portfolio.cash,
                "position": current_position,
                "stock_price": current_price,
                "stock_value": stock_value,
                "total_value": total_value,
                "cumulative_return": cumulative_return,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "position_entry_price": position_entry_price,
                "position_cost_basis": current_position * position_entry_price
                if current_position > 0
                else 0.0,
            }
            portfolio_history.append(portfolio_snapshot)

        return trades, portfolio_history

    def _execute_buy_order(
        self,
        signal: TradingSignal,
        price: float,
        portfolio: Portfolio,
        symbol: str,
        trade_id: int,
        transaction_cost: float = 0.001,
    ) -> Optional[Trade]:
        """
        執行買入訂單

        Args:
            signal: 買入信號
            price: 執行價格
            portfolio: 投資組合狀態
            symbol: 股票代碼
            trade_id: 交易編號

        Returns:
            交易記錄或None（如果無法執行）
        """
        # 只做多模式下，如果已有持股則不重複買入
        if (
            self.config.trading_mode == TradingMode.LONG_ONLY
            and portfolio.get_position(symbol) > 0
        ):
            logger.info(f"已持有 {symbol}，跳過買入信號")
            return None

        # 計算可買入股數
        max_shares = (
            self.config.max_shares_per_trade
            if hasattr(self.config, "max_shares_per_trade")
            else 100
        )
        trade_value = max_shares * price
        commission = trade_value * transaction_cost
        total_cost = trade_value + commission

        # 檢查資金是否足夠
        if portfolio.cash < total_cost:
            # 調整為可負擔的股數
            available_cash = portfolio.cash - commission
            if available_cash <= 0:
                logger.warning(f"資金不足，無法買入 {symbol}")
                return None

            max_shares = int(available_cash // price)
            if max_shares <= 0:
                logger.warning(f"資金不足，無法買入任何 {symbol} 股份")
                return None

            trade_value = max_shares * price
            commission = trade_value * transaction_cost
            total_cost = trade_value + commission

        # 執行買入
        portfolio.cash -= total_cost
        current_position = portfolio.get_position(symbol)
        portfolio.update_position(symbol, current_position + max_shares)

        # 建立交易記錄
        trade = Trade(
            trade_id=f"T{trade_id:04d}",
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type=OrderType.BUY,
            shares=max_shares,
            price=price,
            commission=commission,
            total_cost=total_cost,
            status=TradeStatus.EXECUTED,
            signal_confidence=signal.confidence,
            reason=signal.reason,
        )

        logger.info(f"買入執行: {symbol} {max_shares}股 @ ${price:.2f}")
        return trade

    def _execute_sell_order(
        self,
        signal: TradingSignal,
        price: float,
        portfolio: Portfolio,
        symbol: str,
        trade_id: int,
        transaction_cost: float = 0.001,
    ) -> Optional[Trade]:
        """
        執行賣出訂單

        Args:
            signal: 賣出信號
            price: 執行價格
            portfolio: 投資組合狀態
            symbol: 股票代碼
            trade_id: 交易編號

        Returns:
            交易記錄或None（如果無法執行）
        """
        current_position = portfolio.get_position(symbol)

        # 檢查是否有持股可賣
        if current_position <= 0:
            logger.debug(f"無 {symbol} 持股，無法執行賣出")
            return None

        # 賣出全部持股
        shares_to_sell = current_position
        trade_value = shares_to_sell * price
        commission = trade_value * transaction_cost
        proceeds = trade_value - commission

        # 執行賣出
        portfolio.cash += proceeds
        portfolio.update_position(symbol, 0)

        # 建立交易記錄
        trade = Trade(
            trade_id=f"T{trade_id:04d}",
            timestamp=signal.timestamp,
            symbol=symbol,
            order_type=OrderType.SELL,
            shares=shares_to_sell,
            price=price,
            commission=commission,
            total_cost=proceeds,  # 對賣出而言，這是收入
            status=TradeStatus.EXECUTED,
            signal_confidence=signal.confidence,
            reason=signal.reason,
        )

        logger.info(f"賣出執行: {symbol} {shares_to_sell}股 @ ${price:.2f}")
        return trade

    def _calculate_performance_metrics(
        self,
        portfolio_history: List[Dict[str, Any]],
        data: pd.DataFrame,
        trades: List[Trade],
        signals: List[TradingSignal],
        symbol: str,
        strategy_name: str,
        strategy: Union[TradingStrategy, None] = None,  # 新增策略對象參數
        initial_cash: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        計算績效指標

        Args:
            portfolio_history: 投資組合歷史
            data: 股價數據
            trades: 交易記錄
            signals: 原始交易信號
            symbol: 股票代碼
            strategy_name: 策略名稱

        Returns:
            績效指標字典
        """
        if not portfolio_history:
            raise ValueError("無投資組合歷史數據")

        # 基本資訊
        start_date = portfolio_history[0]["date"]
        end_date = portfolio_history[-1]["date"]
        total_days = len(portfolio_history)

        # 最終數值
        final_value = portfolio_history[-1]["total_value"]
        final_return = portfolio_history[-1]["cumulative_return"]

        # 計算年化報酬率
        days_in_year = 365.25
        years = total_days / days_in_year
        annual_return = (
            (final_value / self.config.initial_capital) ** (1 / years) - 1
            if years > 0
            else 0
        )

        # 計算波動率
        returns = [ph["cumulative_return"] for ph in portfolio_history]
        returns_series = pd.Series(returns)
        daily_returns = returns_series.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)  # 年化波動率

        # 處理 NaN 值
        if pd.isna(volatility):
            volatility = 0.0

        # 計算最大回撤
        values = [ph["total_value"] for ph in portfolio_history]
        cummax = pd.Series(values).cummax()
        drawdown = (pd.Series(values) - cummax) / cummax
        max_drawdown = drawdown.min()

        # 處理 NaN 值
        if pd.isna(max_drawdown):
            max_drawdown = 0.0

        # 交易統計
        num_trades = len(trades)
        buy_trades = [t for t in trades if t.order_type == OrderType.BUY]
        sell_trades = [t for t in trades if t.order_type == OrderType.SELL]

        # 計算勝率（需要配對買賣交易）
        win_rate = 0.0
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            # 簡化勝率計算：比較買入和賣出價格
            paired_trades = min(len(buy_trades), len(sell_trades))
            wins = sum(
                1
                for i in range(paired_trades)
                if sell_trades[i].price > buy_trades[i].price
            )
            win_rate = wins / paired_trades if paired_trades > 0 else 0

        # 基準比較（買入持有）
        buy_hold_return = (data["close"].iloc[-1] / data["close"].iloc[0]) - 1
        alpha = final_return - buy_hold_return

        # 生成交易事件
        trading_events = self._generate_trading_events(
            trades, portfolio_history, symbol
        )

        # 轉換原始信號為前端可用格式
        trading_signals = []
        for signal in signals:
            signal_type_str = (
                signal.signal_type.name
                if hasattr(signal.signal_type, "name")
                else str(signal.signal_type)
            )
            trading_signals.append(
                {
                    "timestamp": signal.timestamp.isoformat()
                    if hasattr(signal.timestamp, "isoformat")
                    else str(signal.timestamp),
                    "signal_type": signal_type_str.upper(),
                    "confidence": float(signal.confidence),
                    "price": float(signal.price) if signal.price else 0.0,
                    "reason": str(signal.reason) if signal.reason else "",
                    "metadata": signal.metadata or {},
                }
            )

        # 整合結果，確保所有數值都是 Python 原生類型
        results = {
            "basic_info": {
                "symbol": str(symbol),
                "strategy_name": str(strategy_name),
                "start_date": start_date.isoformat()
                if hasattr(start_date, "isoformat")
                else str(start_date),
                "end_date": end_date.isoformat()
                if hasattr(end_date, "isoformat")
                else str(end_date),
                "total_days": int(total_days),
                "initial_capital": float(initial_cash),
                "max_shares_per_trade": int(self.config.max_shares_per_trade),
            },
            "performance_metrics": {
                "final_value": float(final_value),
                "total_return": float(final_return),
                "annual_return": float(annual_return),
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "num_trades": int(num_trades),
                "win_rate": float(win_rate),
            },
            "strategy_statistics": {
                # 從策略中獲取詳細統計（如果可用）
                "total_realized_pnl": getattr(strategy, "total_realized_pnl", 0.0),
                "total_trades": getattr(strategy, "total_trades", num_trades),
                "winning_trades": getattr(strategy, "winning_trades", 0),
                "strategy_win_rate": (
                    getattr(strategy, "winning_trades", 0)
                    / getattr(strategy, "total_trades", 1)
                )
                if getattr(strategy, "total_trades", 0) > 0
                else 0.0,
                "cumulative_trade_return_rate": sum(
                    getattr(strategy, "trade_returns", [])
                )
                / 100
                if hasattr(strategy, "trade_returns") and strategy.trade_returns
                else 0.0,
            },
            "benchmark_comparison": {
                "buy_hold_return": float(buy_hold_return),
                "strategy_return": float(final_return),
                "alpha": float(alpha),
                "outperformed": bool(alpha > 0),
            },
            "trades": [trade.to_dict() for trade in trades],
            "trading_signals": trading_signals,  # 新增：原始交易信號
            "portfolio_history": portfolio_history,
            "trading_events": trading_events,
            "stock_data": [
                {
                    "timestamp": idx.isoformat()
                    if hasattr(idx, "isoformat")
                    else str(idx),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]) if not pd.isna(row["volume"]) else 0,
                }
                for idx, row in data.iterrows()
            ],
        }

        return results

    def _generate_trading_events(
        self, trades: List[Trade], portfolio_history: List[Dict[str, Any]], symbol: str
    ) -> List[Dict[str, Any]]:
        """
        生成詳細的交易事件日誌

        Args:
            trades: 交易記錄列表
            portfolio_history: 投資組合歷史
            symbol: 股票代碼

        Returns:
            交易事件列表
        """
        events = []

        # 建立日期到投資組合狀態的映射
        portfolio_dict = {}
        for ph in portfolio_history:
            date_key = ph["date"].date() if hasattr(ph["date"], "date") else ph["date"]
            portfolio_dict[date_key] = ph

        # 處理每筆交易
        for trade in trades:
            trade_date = (
                trade.timestamp.date()
                if hasattr(trade.timestamp, "date")
                else trade.timestamp
            )
            portfolio_state = portfolio_dict.get(trade_date, {})

            if trade.order_type == OrderType.BUY:
                event_type = "buy_success"
                description = f"買入信號執行成功，買入 {symbol} {trade.shares}股，價格 ${trade.price:.2f}"
            else:
                event_type = "sell_success"
                description = f"賣出信號執行成功，賣出 {symbol} {trade.shares}股，價格 ${trade.price:.2f}"

            events.append(
                {
                    "date": trade.timestamp.isoformat()
                    if hasattr(trade.timestamp, "isoformat")
                    else str(trade.timestamp),
                    "event_type": event_type,
                    "signal_type": trade.order_type.value.upper(),
                    "signal_confidence": trade.signal_confidence,
                    "execution_price": trade.price,
                    "shares_traded": trade.shares,
                    "trade_amount": trade.total_cost,
                    "commission": trade.commission,
                    "current_position": portfolio_state.get("position", 0),
                    "current_cash": portfolio_state.get("cash", 0),
                    "current_equity": portfolio_state.get("total_value", 0),
                    "cumulative_return": portfolio_state.get("cumulative_return", 0),
                    "description": description,
                }
            )

        # 添加最終結算事件
        if portfolio_history:
            final_state = portfolio_history[-1]
            events.append(
                {
                    "date": final_state["date"].isoformat()
                    if hasattr(final_state["date"], "isoformat")
                    else str(final_state["date"]),
                    "event_type": "final_settlement",
                    "signal_type": "SETTLEMENT",
                    "current_position": final_state["position"],
                    "current_cash": final_state["cash"],
                    "stock_value": final_state["stock_value"],
                    "current_equity": final_state["total_value"],
                    "cumulative_return": final_state["cumulative_return"],
                    "description": f"最終結算 - 持有{symbol} {final_state['position']}股，現金 ${final_state['cash']:,.0f}，總資產 ${final_state['total_value']:,.0f}，累積報酬率 {final_state['cumulative_return'] * 100:.2f}%",
                }
            )

        return events

    def get_backtest_chart(
        self, symbol: str, strategy_name: str, show_trades: bool = True
    ) -> go.Figure:
        """
        生成回測結果圖表

        Args:
            symbol: 交易標的
            strategy_name: 策略名稱
            show_trades: 是否顯示交易點

        Returns:
            Plotly 圖表物件
        """
        result_key = f"{symbol}_{strategy_name}"

        if result_key not in self.current_results:
            raise ValueError(f"找不到 {symbol} 策略 {strategy_name} 的回測結果")

        result = self.current_results[result_key]

        # 創建子圖
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[f"{symbol} 價格與交易", "累積報酬率"],
            row_heights=[0.7, 0.3],
        )

        # 股價數據
        stock_data = result["stock_data"]
        dates = [pd.to_datetime(d["timestamp"]) for d in stock_data]

        # 添加股價蜡燭圖
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=[d["open"] for d in stock_data],
                high=[d["high"] for d in stock_data],
                low=[d["low"] for d in stock_data],
                close=[d["close"] for d in stock_data],
                name=f"{symbol} 價格",
            ),
            row=1,
            col=1,
        )

        # 添加交易點
        if show_trades:
            buy_trades = [t for t in result["trades"] if t["order_type"] == "buy"]
            sell_trades = [t for t in result["trades"] if t["order_type"] == "sell"]

            if buy_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(t["timestamp"]) for t in buy_trades],
                        y=[t["price"] for t in buy_trades],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="green"),
                        name="買入",
                        text=[f"買入 {t['shares']}股" for t in buy_trades],
                        hovertemplate="%{text}<br>價格: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            if sell_trades:
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(t["timestamp"]) for t in sell_trades],
                        y=[t["price"] for t in sell_trades],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="red"),
                        name="賣出",
                        text=[f"賣出 {t['shares']}股" for t in sell_trades],
                        hovertemplate="%{text}<br>價格: $%{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # 添加累積報酬率曲線
        portfolio_history = result["portfolio_history"]
        portfolio_dates = [pd.to_datetime(ph["date"]) for ph in portfolio_history]
        cumulative_returns = [ph["cumulative_return"] * 100 for ph in portfolio_history]

        fig.add_trace(
            go.Scatter(
                x=portfolio_dates,
                y=cumulative_returns,
                mode="lines",
                name="策略累積報酬率",
                line=dict(color="blue"),
            ),
            row=2,
            col=1,
        )

        # 添加基準線（買入持有）
        buy_hold_return = result["benchmark_comparison"]["buy_hold_return"]
        fig.add_hline(
            y=buy_hold_return * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"買入持有: {buy_hold_return * 100:.2f}%",
            row=2,
            col=1,
        )

        # 更新佈局
        fig.update_layout(
            title=f"回測結果: {symbol} - {strategy_name}",
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False,
        )

        fig.update_yaxes(title_text="價格 ($)", row=1, col=1)
        fig.update_yaxes(title_text="累積報酬率 (%)", row=2, col=1)
        fig.update_xaxes(title_text="日期", row=2, col=1)

        return fig

    def get_backtest_summary(self, symbol: str, strategy_name: str) -> str:
        """
        獲取回測摘要文字

        Args:
            symbol: 交易標的
            strategy_name: 策略名稱

        Returns:
            格式化的回測摘要
        """
        result_key = f"{symbol}_{strategy_name}"

        if result_key not in self.current_results:
            return f"找不到 {symbol} 策略 {strategy_name} 的回測結果"

        result = self.current_results[result_key]
        basic = result["basic_info"]
        metrics = result["performance_metrics"]
        benchmark = result["benchmark_comparison"]

        summary = f"""
回測摘要報告
{"=" * 60}

基礎資訊:
- 交易標的: {basic["symbol"]}
- 策略名稱: {basic["strategy_name"]}
- 回測期間: {basic["start_date"]} 至 {basic["end_date"]}
- 交易天數: {basic["total_days"]}
- 初始資金: ${basic["initial_capital"]:,.0f}
- 每次最大買入股數: {basic["max_shares_per_trade"]}

績效表現:
- 最終資產: ${metrics["final_value"]:,.0f}
- 總報酬率: {metrics["total_return"] * 100:.2f}%
- 年化報酬率: {metrics["annual_return"] * 100:.2f}%
- 年化波動率: {metrics["volatility"] * 100:.2f}%
- 最大回撤: {metrics["max_drawdown"] * 100:.2f}%

交易統計:
- 交易次數: {metrics["num_trades"]}
- 勝率: {metrics["win_rate"] * 100:.2f}%

基準比較:
- 買入持有報酬: {benchmark["buy_hold_return"] * 100:.2f}%
- 策略超額報酬: {benchmark["alpha"] * 100:.2f}%
- 是否跑贏基準: {"是" if benchmark["outperformed"] else "否"}
        """.strip()

        return summary


# 為了向後相容，保留原始類別名稱的別名
BacktestEngine = CustomBacktestEngine
