"""
Unrealized P&L Tracker
For calculating and displaying unrealized profit and loss of current positions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


class UnrealizedPnLTracker:
    """Unrealized P&L Tracker"""

    def __init__(self):
        self.positions = []  # Position records

    def add_position(
        self,
        symbol: str,
        entry_date: str,
        entry_price: float,
        quantity: int = 1,
        signal_confidence: float = 1.0,
    ):
        """添加持倉記錄"""
        position = {
            "symbol": symbol,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "quantity": quantity,
            "signal_confidence": signal_confidence,
            "entry_timestamp": datetime.now(),
        }
        self.positions.append(position)
        return len(self.positions) - 1  # 返回持倉ID

    def calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> List[Dict]:
        """計算所有持倉的未實現損益"""
        results = []

        for i, position in enumerate(self.positions):
            symbol = position["symbol"]
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position["entry_price"]
            quantity = position["quantity"]

            # 計算損益
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_return = (current_price - entry_price) / entry_price
            unrealized_pnl_percent = unrealized_return * 100

            result = {
                "position_id": i,
                "symbol": symbol,
                "entry_date": position["entry_date"],
                "entry_price": entry_price,
                "current_price": current_price,
                "quantity": quantity,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_return": unrealized_return,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "signal_confidence": position["signal_confidence"],
                "holding_days": (datetime.now() - position["entry_timestamp"]).days,
            }
            results.append(result)

        return results

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """獲取投資組合未實現損益總結"""
        pnl_results = self.calculate_unrealized_pnl(current_prices)

        if not pnl_results:
            return {"error": "無持倉記錄"}

        total_unrealized_pnl = sum(r["unrealized_pnl"] for r in pnl_results)
        total_investment = sum(r["entry_price"] * r["quantity"] for r in pnl_results)
        portfolio_return = (
            total_unrealized_pnl / total_investment if total_investment > 0 else 0
        )

        # 加權平均未實現損益（按信心度加權）
        weighted_return = 0
        total_weight = sum(r["signal_confidence"] for r in pnl_results)
        if total_weight > 0:
            weighted_return = (
                sum(
                    r["unrealized_pnl_percent"] * r["signal_confidence"]
                    for r in pnl_results
                )
                / total_weight
            )

        return {
            "total_positions": len(pnl_results),
            "total_investment": total_investment,
            "total_unrealized_pnl": total_unrealized_pnl,
            "portfolio_return_percent": portfolio_return * 100,
            "weighted_avg_return_percent": weighted_return,
            "best_position": max(
                pnl_results, key=lambda x: x["unrealized_pnl_percent"]
            ),
            "worst_position": min(
                pnl_results, key=lambda x: x["unrealized_pnl_percent"]
            ),
            "positions": pnl_results,
        }

    def close_position(self, position_id: int, exit_price: float, exit_date: str):
        """平倉（移除持倉記錄）"""
        if 0 <= position_id < len(self.positions):
            position = self.positions.pop(position_id)

            # 計算實現損益
            realized_pnl = (exit_price - position["entry_price"]) * position["quantity"]
            realized_return = (exit_price - position["entry_price"]) / position[
                "entry_price"
            ]

            return {
                "symbol": position["symbol"],
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "quantity": position["quantity"],
                "realized_pnl": realized_pnl,
                "realized_return_percent": realized_return * 100,
                "holding_days": (
                    datetime.strptime(exit_date, "%Y-%m-%d")
                    - datetime.strptime(position["entry_date"], "%Y-%m-%d")
                ).days,
            }
        return None


# 使用示例
def demo_unrealized_pnl():
    """演示未實現損益功能"""
    tracker = UnrealizedPnLTracker()

    # 模擬添加幾個持倉
    tracker.add_position("2330.TW", "2024-12-01", 980.0, 100, 0.75)
    tracker.add_position("2330.TW", "2025-01-15", 1050.0, 50, 0.68)
    tracker.add_position("TSLA", "2024-11-20", 350.0, 10, 0.82)

    # 模擬當前價格
    current_prices = {"2330.TW": 1080.0, "TSLA": 380.0}

    # 計算未實現損益
    summary = tracker.get_portfolio_summary(current_prices)

    print("📊 投資組合未實現損益總結:")
    print(f"  總持倉數: {summary['total_positions']}")
    print(f"  總投資金額: ${summary['total_investment']:,.2f}")
    print(f"  總未實現損益: ${summary['total_unrealized_pnl']:+,.2f}")
    print(f"  投資組合報酬率: {summary['portfolio_return_percent']:+.2f}%")
    print(f"  加權平均報酬率: {summary['weighted_avg_return_percent']:+.2f}%")

    print(f"\n📈 個別持倉:")
    for pos in summary["positions"]:
        print(
            f"  {pos['symbol']}: 進場@${pos['entry_price']:.2f}, "
            f"現價@${pos['current_price']:.2f}, "
            f"未實現損益: {pos['unrealized_pnl_percent']:+.2f}%"
        )


if __name__ == "__main__":
    demo_unrealized_pnl()
