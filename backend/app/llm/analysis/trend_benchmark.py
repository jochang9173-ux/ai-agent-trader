"""
Trend Analysis Benchmark Database
Used for validating and improving the accuracy of trend identification algorithms
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Tuple

import pandas as pd


@dataclass
class TrendPhase:
    """趨勢階段定義"""

    start_date: str
    end_date: str
    trend_type: str  # 'uptrend', 'downtrend', 'sideways', 'sideways_bullish', 'sideways_bearish'
    strength: str  # 'strong', 'moderate', 'weak'
    confidence: float  # 0.0 - 1.0 人工標註的信心度
    description: str  # 描述性說明
    key_events: List[str] = None  # 關鍵事件或轉折點


@dataclass
class BenchmarkCase:
    """基準測試案例"""

    symbol: str
    period: str
    total_period: Tuple[str, str]  # (start_date, end_date)
    phases: List[TrendPhase]
    overall_trend: str
    complexity_level: str  # 'simple', 'moderate', 'complex'
    notes: str


class TrendBenchmarkDatabase:
    """趨勢分析基準數據庫"""

    def __init__(self):
        self.benchmark_cases = self._initialize_benchmark_cases()

    def _initialize_benchmark_cases(self) -> Dict[str, BenchmarkCase]:
        """初始化基準測試案例"""
        cases = {}

        # 2330.TW 案例 - 複雜多階段趨勢
        tsmc_2024_2025 = BenchmarkCase(
            symbol="2330.TW",
            period="近一年 (2024年10月-2025年7月)",
            total_period=("2024-10-01", "2025-07-30"),
            phases=[
                TrendPhase(
                    start_date="2024-10-20",
                    end_date="2024-11-14",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="區間震盪，價格在特定範圍內波動，無明顯方向性",
                    key_events=["橫盤整理", "等待突破方向"],
                ),
                TrendPhase(
                    start_date="2025-02-20",
                    end_date="2025-04-20",
                    trend_type="downtrend",
                    strength="moderate",
                    confidence=0.85,
                    description="明顯下降趨勢，低點逐步降低",
                    key_events=["突破支撐位", "持續下跌"],
                ),
                TrendPhase(
                    start_date="2025-04-23",
                    end_date="2025-07-23",
                    trend_type="uptrend",
                    strength="moderate",
                    confidence=0.9,
                    description="上升趨勢，高點和低點逐步抬升",
                    key_events=["突破阻力位", "持續上漲"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="典型的多階段趨勢轉換案例，包含震盪->下跌->上升的完整週期",
        )
        cases["TSMC_2024_2025"] = tsmc_2024_2025

        # TSLA 案例 - 下降趨勢 + 兩個區間震盪階段
        tsla_2025 = BenchmarkCase(
            symbol="TSLA",
            period="近六個月 (2025年1月-7月)",
            total_period=("2025-01-22", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-01-22",
                    end_date="2025-03-11",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="明顯的下降趨勢階段",
                    key_events=["市場調整", "獲利了結"],
                ),
                TrendPhase(
                    start_date="2025-03-11",
                    end_date="2025-05-05",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="第一個區間震盪階段，價格在一定範圍內波動",
                    key_events=["市場穩定", "橫盤整理"],
                ),
                TrendPhase(
                    start_date="2025-05-28",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.8,
                    description="第二個區間震盪階段，繼續橫盤整理",
                    key_events=["持續整理", "等待突破"],
                ),
            ],
            overall_trend="complex_multi_phase",
            complexity_level="complex",
            notes="包含下降趨勢和兩個區間震盪階段的複雜案例，測試區間識別能力",
        )
        cases["TSLA_2025"] = tsla_2025

        # AAPL 案例 - 下降趨勢轉震盪的兩階段模式
        aapl_2025 = BenchmarkCase(
            symbol="AAPL",
            period="2025年2月-7月",
            total_period=("2025-02-01", "2025-07-28"),
            phases=[
                TrendPhase(
                    start_date="2025-02-25",
                    end_date="2025-04-08",
                    trend_type="downtrend",
                    strength="strong",
                    confidence=0.9,
                    description="明確的下降趨勢階段，價格從高點大幅回落",
                    key_events=["市場回調", "技術性調整", "從 $246.72 跌至 $172.19"],
                ),
                TrendPhase(
                    start_date="2025-04-09",
                    end_date="2025-07-28",
                    trend_type="sideways",
                    strength="moderate",
                    confidence=0.85,
                    description="震盪整理階段，價格在區間內來回波動",
                    key_events=["橫盤整理", "12.7% 區間震盪", "多次測試支撐阻力"],
                ),
            ],
            overall_trend="trend_to_sideways_transition",
            complexity_level="medium",
            notes="典型的趨勢轉震盪案例，適合測試算法對趨勢轉換點的識別能力和區間震盪檢測",
        )
        cases["AAPL_2025"] = aapl_2025

        return cases

    def get_benchmark_case(self, case_id: str) -> BenchmarkCase:
        """獲取特定基準案例"""
        return self.benchmark_cases.get(case_id)

    def add_benchmark_case(self, case_id: str, benchmark_case: BenchmarkCase):
        """添加新的基準案例"""
        self.benchmark_cases[case_id] = benchmark_case

    def get_all_cases(self) -> Dict[str, BenchmarkCase]:
        """獲取所有基準案例"""
        return self.benchmark_cases

    def evaluate_algorithm_performance(
        self, case_id: str, algorithm_result: Dict
    ) -> Dict[str, Any]:
        """評估算法在特定案例上的表現"""
        benchmark = self.get_benchmark_case(case_id)
        if not benchmark:
            return {"error": f"找不到基準案例: {case_id}"}

        evaluation = {
            "case_id": case_id,
            "benchmark_phases": len(benchmark.phases),
            "detected_phases": algorithm_result.get("detected_phases", 0),
            "overall_accuracy": 0.0,
            "phase_accuracy": [],
            "missed_transitions": [],
            "false_positives": [],
        }

        # 這裡可以實現具體的評估邏輯
        # 比較算法檢測的趨勢階段與人工標註的基準

        return evaluation


# 全局實例
trend_benchmark_db = TrendBenchmarkDatabase()


def get_benchmark_database() -> TrendBenchmarkDatabase:
    """獲取基準數據庫實例"""
    return trend_benchmark_db


def create_test_case_from_data(
    symbol: str, market_data: List[Dict], manual_phases: List[Dict]
) -> BenchmarkCase:
    """從市場數據和人工標註創建測試案例"""
    phases = []
    for phase_data in manual_phases:
        phase = TrendPhase(
            start_date=phase_data["start_date"],
            end_date=phase_data["end_date"],
            trend_type=phase_data["trend_type"],
            strength=phase_data.get("strength", "moderate"),
            confidence=phase_data.get("confidence", 0.8),
            description=phase_data.get("description", ""),
            key_events=phase_data.get("key_events", []),
        )
        phases.append(phase)

    # 確定整體複雜度
    complexity = "simple"
    if len(phases) > 2:
        complexity = "moderate"
    if len(phases) > 3 or any(p.trend_type.startswith("sideways") for p in phases):
        complexity = "complex"

    return BenchmarkCase(
        symbol=symbol,
        period=f"{phases[0].start_date} to {phases[-1].end_date}",
        total_period=(phases[0].start_date, phases[-1].end_date),
        phases=phases,
        overall_trend="multi_phase" if len(phases) > 1 else phases[0].trend_type,
        complexity_level=complexity,
        notes=f"自動生成的測試案例，包含 {len(phases)} 個趨勢階段",
    )
