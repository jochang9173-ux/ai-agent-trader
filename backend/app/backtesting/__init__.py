"""
Backtesting Engine Core Module

Provides complete strategy backtesting functionality, including portfolio management,
trade execution simulation, and performance metrics calculation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..llm.strategies import SignalType, TradingSignal, TradingStrategy


class OrderType(Enum):
    """Order Types"""

    MARKET = "market"  # Market order
    LIMIT = "limit"  # Limit order
    STOP = "stop"  # Stop order
    STOP_LIMIT = "stop_limit"  # Stop limit order


class OrderStatus(Enum):
    """Order Status"""

    PENDING = "pending"  # 待执行
    FILLED = "filled"  # 已成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"  # 已拒绝


@dataclass
class Order:
    """
    订单数据类

    Attributes:
        order_id: 订单ID
        symbol: 交易标的
        order_type: 订单类型
        side: 买卖方向 (BUY/SELL)
        quantity: 交易数量
        price: 订单价格 (市价单为None)
        timestamp: 下单时间
        status: 订单状态
        filled_price: 成交价格
        filled_quantity: 成交数量
        commission: 手续费
        metadata: 额外信息
    """

    order_id: str
    symbol: str
    order_type: OrderType
    side: SignalType
    quantity: float
    price: Optional[float]
    timestamp: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """
    持仓数据类

    Attributes:
        symbol: 交易标的
        quantity: 持仓数量 (正数为多头，负数为空头)
        avg_cost: 平均成本
        market_value: 市值
        unrealized_pnl: 未实现盈亏
        realized_pnl: 已实现盈亏
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_market_value(self, current_price: float) -> None:
        """更新市值和未实现盈亏"""
        self.market_value = self.quantity * current_price
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.avg_cost) * self.quantity


@dataclass
class BacktestConfig:
    """
    回测配置类

    Attributes:
        initial_capital: 初始资金
        commission_rate: 手续费率 (0.001 = 0.1%)
        slippage_rate: 滑点率 (0.0005 = 0.05%)
        min_trade_amount: 最小交易金额
        max_position_size: 最大持仓比例 (0.5 = 50%)
        enable_short_selling: 是否允许做空
        margin_requirement: 保证金要求 (做空时使用)
        risk_free_rate: 无风险利率 (年化)
    """

    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_trade_amount: float = 100.0
    max_position_size: float = 1.0
    enable_short_selling: bool = False
    margin_requirement: float = 0.5
    risk_free_rate: float = 0.02


class Portfolio:
    """
    投资组合管理器

    管理现金、持仓、订单历史等投资组合状态
    """

    def __init__(self, config: BacktestConfig):
        """
        初始化投资组合

        Args:
            config: 回测配置
        """
        self.config = config
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders_history: List[Order] = []
        self.daily_returns: List[float] = []
        self.portfolio_values: List[Tuple[pd.Timestamp, float]] = []

        # 性能追踪
        self.total_commission_paid = 0.0
        self.total_slippage_cost = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = config.initial_capital

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        计算当前投资组合总值

        Args:
            current_prices: 当前价格字典 {symbol: price}

        Returns:
            投资组合总值
        """
        total_value = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_market_value(current_prices[symbol])
                total_value += position.market_value

        return total_value

    def get_available_cash(self) -> float:
        """获取可用现金"""
        return self.cash

    def get_position(self, symbol: str) -> Position:
        """
        获取持仓信息

        Args:
            symbol: 交易标的

        Returns:
            持仓对象
        """
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def calculate_position_size(
        self, symbol: str, signal: TradingSignal, current_price: float
    ) -> float:
        """
        计算交易数量

        Args:
            symbol: 交易标的
            signal: 交易信号
            current_price: 当前价格

        Returns:
            建议交易数量
        """
        portfolio_value = self.get_portfolio_value({symbol: current_price})

        # 基于置信度和最大持仓比例计算仓位
        confidence_factor = signal.confidence
        max_position_value = (
            portfolio_value * self.config.max_position_size * confidence_factor
        )

        if signal.signal_type == SignalType.BUY:
            # 买入信号
            available_cash = self.get_available_cash()
            trade_value = min(max_position_value, available_cash * 0.95)  # 保留5%现金

            if trade_value < self.config.min_trade_amount:
                return 0.0

            # 考虑手续费和滑点
            effective_price = current_price * (
                1 + self.config.commission_rate + self.config.slippage_rate
            )
            quantity = trade_value / effective_price

        elif signal.signal_type == SignalType.SELL:
            # 卖出信号
            current_position = self.get_position(symbol)
            if current_position.quantity <= 0:
                return 0.0

            # 基于置信度决定卖出比例
            sell_ratio = confidence_factor
            quantity = current_position.quantity * sell_ratio

        else:
            return 0.0

        return max(0, quantity)

    def place_order(
        self,
        symbol: str,
        side: SignalType,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Order:
        """
        下单

        Args:
            symbol: 交易标的
            side: 买卖方向
            quantity: 交易数量
            order_type: 订单类型
            price: 限价单价格
            timestamp: 下单时间

        Returns:
            订单对象
        """
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        order_id = f"{symbol}_{side.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        order = Order(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
        )

        self.orders_history.append(order)
        return order

    def execute_order(self, order: Order, market_price: float) -> bool:
        """
        执行订单

        Args:
            order: 订单对象
            market_price: 市场价格

        Returns:
            是否执行成功
        """
        if order.status != OrderStatus.PENDING:
            return False

        # 计算实际成交价格 (考虑滑点)
        if order.side == SignalType.BUY:
            slippage = market_price * self.config.slippage_rate
            filled_price = market_price + slippage
        else:  # SELL
            slippage = market_price * self.config.slippage_rate
            filled_price = market_price - slippage

        # 计算手续费
        trade_value = order.quantity * filled_price
        commission = trade_value * self.config.commission_rate

        # 检查是否有足够资金
        if order.side == SignalType.BUY:
            required_cash = trade_value + commission
            if required_cash > self.cash:
                order.status = OrderStatus.REJECTED
                order.metadata["rejection_reason"] = "Insufficient funds"
                return False

        # 执行交易
        position = self.get_position(order.symbol)

        if order.side == SignalType.BUY:
            # 买入
            total_cost = position.quantity * position.avg_cost
            new_total_cost = total_cost + trade_value
            new_quantity = position.quantity + order.quantity

            if new_quantity > 0:
                position.avg_cost = new_total_cost / new_quantity
            position.quantity = new_quantity

            self.cash -= trade_value + commission

        else:  # SELL
            # 卖出
            if order.quantity > position.quantity:
                # 不能卖出超过持仓数量
                order.quantity = position.quantity
                trade_value = order.quantity * filled_price
                commission = trade_value * self.config.commission_rate

            # 计算已实现盈亏
            realized_pnl = (filled_price - position.avg_cost) * order.quantity
            position.realized_pnl += realized_pnl
            position.quantity -= order.quantity

            self.cash += trade_value - commission

        # 更新订单状态
        order.status = OrderStatus.FILLED
        order.filled_price = filled_price
        order.filled_quantity = order.quantity
        order.commission = commission

        # 更新统计
        self.total_commission_paid += commission
        self.total_slippage_cost += abs(filled_price - market_price) * order.quantity

        return True

    def update_portfolio_value(
        self, timestamp: pd.Timestamp, current_prices: Dict[str, float]
    ) -> None:
        """
        更新投资组合价值记录

        Args:
            timestamp: 时间戳
            current_prices: 当前价格
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        self.portfolio_values.append((timestamp, portfolio_value))

        # 更新最大回撤
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        else:
            current_drawdown = (
                self.peak_portfolio_value - portfolio_value
            ) / self.peak_portfolio_value
            self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # 计算日收益率
        if len(self.portfolio_values) > 1:
            prev_value = self.portfolio_values[-2][1]
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        获取投资组合摘要

        Returns:
            投资组合摘要字典
        """
        if not self.portfolio_values:
            return {}

        current_value = self.portfolio_values[-1][1]
        total_return = (current_value - self.initial_capital) / self.initial_capital

        return {
            "initial_capital": self.initial_capital,
            "current_value": current_value,
            "cash": self.cash,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown * 100,
            "total_commission": self.total_commission_paid,
            "total_slippage": self.total_slippage_cost,
            "num_trades": len(
                [o for o in self.orders_history if o.status == OrderStatus.FILLED]
            ),
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                }
                for symbol, pos in self.positions.items()
                if pos.quantity != 0
            },
        }
