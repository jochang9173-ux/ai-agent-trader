"""
Backtest Logger
Records detailed decision processes and analysis data for LLM strategies
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class BacktestLogger:
    """
    Backtest Logger

    Records detailed decision processes for LLM strategies, including:
    - Daily market analysis
    - Triggered events
    - LLM decision processes
    - Trading signals
    - Strategy states
    """

    def __init__(
        self, db_path: str = "backend/data/backtest_logs.db", session_id: str = None
    ):
        """
        Initialize the logger

        Args:
            db_path: SQLite database file path
            session_id: Backtest session ID, auto-generated if not provided
        """
        self.db_path = Path(db_path)
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or str(uuid.uuid4())
        self._init_database()

    def _init_database(self):
        """初始化數據庫結構"""
        with sqlite3.connect(self.db_path) as conn:
            # 創建主表：每日分析日誌
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    
                    -- 基本市場數據 (結構化，便於查詢)
                    price REAL,
                    volume INTEGER,
                    daily_return REAL,
                    volatility REAL,
                    
                    -- 趨勢分析 (JSON)
                    trend_analysis TEXT, -- JSON字符串
                    
                    -- 全面技術分析 (JSON) - 新增
                    comprehensive_technical_analysis TEXT, -- JSON字符串
                    
                    -- 觸發事件 (JSON)
                    triggered_events TEXT, -- JSON字符串
                    
                    -- LLM決策 (JSON)
                    llm_decision TEXT, -- JSON字符串
                    
                    -- 交易信號 (JSON)
                    trading_signal TEXT, -- JSON字符串
                    
                    -- 策略狀態 (JSON)
                    strategy_state TEXT, -- JSON字符串
                    
                    -- 結果評估 (後續更新)
                    actual_pnl REAL,
                    prediction_accuracy REAL,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 創建事件分析表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_analysis_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    daily_log_id INTEGER,
                    event_type TEXT NOT NULL,
                    severity TEXT,
                    detection_time DATETIME,
                    
                    -- 市場上下文 (JSON)
                    market_context TEXT, -- JSON字符串
                    
                    -- LLM響應 (JSON) 
                    llm_response TEXT, -- JSON字符串
                    
                    -- 效果評估 (JSON)
                    effectiveness TEXT, -- JSON字符串
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (daily_log_id) REFERENCES daily_analysis_logs (id)
                )
            """)

            # 創建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_logs_date_symbol 
                ON daily_analysis_logs (date, symbol)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_logs_session 
                ON daily_analysis_logs (session_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type 
                ON event_analysis_logs (event_type)
            """)

            conn.commit()

    def log_daily_analysis(
        self,
        symbol: str,
        date: str,
        market_data: Dict[str, Any],
        trend_analysis: Dict[str, Any] = None,
        comprehensive_technical_analysis: Dict[str, Any] = None,  # 新增參數
        triggered_events: List[Dict[str, Any]] = None,
        llm_decision: Dict[str, Any] = None,
        trading_signal: Dict[str, Any] = None,
        strategy_state: Dict[str, Any] = None,
    ) -> int:
        """
        記錄每日分析數據 (新記錄會覆蓋同一股票同一天的舊記錄)

        Args:
            symbol: 股票代碼
            date: 日期 (YYYY-MM-DD)
            market_data: 市場數據字典
            trend_analysis: 趨勢分析結果
            comprehensive_technical_analysis: 全面技術分析結果
            triggered_events: 觸發事件列表
            llm_decision: LLM決策結果
            trading_signal: 交易信號
            strategy_state: 策略狀態

        Returns:
            記錄的ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 檢查是否存在相同的記錄 (同一symbol + date)
            cursor.execute(
                """
                SELECT id FROM daily_analysis_logs 
                WHERE symbol = ? AND date = ?
                ORDER BY timestamp DESC
            """,
                (symbol, date),
            )

            existing_records = cursor.fetchall()

            if existing_records:
                # 刪除舊記錄和相關的事件記錄
                old_ids = [record[0] for record in existing_records]
                old_ids_str = ",".join("?" * len(old_ids))

                # 先刪除相關的事件分析記錄
                cursor.execute(
                    f"""
                    DELETE FROM event_analysis_logs 
                    WHERE daily_log_id IN ({old_ids_str})
                """,
                    old_ids,
                )

                # 再刪除每日分析記錄
                cursor.execute(
                    f"""
                    DELETE FROM daily_analysis_logs 
                    WHERE id IN ({old_ids_str})
                """,
                    old_ids,
                )

                print(f"🔄 覆蓋 {symbol} - {date} 的舊記錄 ({len(old_ids)}條)")

            # 插入新記錄
            cursor.execute(
                """
                INSERT INTO daily_analysis_logs (
                    session_id, symbol, date, timestamp,
                    price, volume, daily_return, volatility,
                    trend_analysis, comprehensive_technical_analysis, triggered_events, llm_decision,
                    trading_signal, strategy_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.session_id,
                    symbol,
                    date,
                    datetime.now().isoformat(),
                    market_data.get("price"),
                    market_data.get("volume"),
                    market_data.get("daily_return"),
                    market_data.get("volatility"),
                    json.dumps(trend_analysis) if trend_analysis else None,
                    json.dumps(comprehensive_technical_analysis)
                    if comprehensive_technical_analysis
                    else None,
                    json.dumps(triggered_events) if triggered_events else None,
                    json.dumps(llm_decision) if llm_decision else None,
                    json.dumps(trading_signal) if trading_signal else None,
                    json.dumps(strategy_state) if strategy_state else None,
                ),
            )

            return cursor.lastrowid

    def log_event_analysis(
        self,
        daily_log_id: int,
        event_type: str,
        severity: str,
        market_context: Dict[str, Any] = None,
        llm_response: Dict[str, Any] = None,
        effectiveness: Dict[str, Any] = None,
    ):
        """
        記錄事件分析數據

        Args:
            daily_log_id: 對應的日誌記錄ID
            event_type: 事件類型
            severity: 嚴重程度
            market_context: 市場上下文
            llm_response: LLM響應
            effectiveness: 效果評估
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO event_analysis_logs (
                    session_id, daily_log_id, event_type, severity,
                    detection_time, market_context, llm_response, effectiveness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.session_id,
                    daily_log_id,
                    event_type,
                    severity,
                    datetime.now().isoformat(),
                    json.dumps(market_context) if market_context else None,
                    json.dumps(llm_response) if llm_response else None,
                    json.dumps(effectiveness) if effectiveness else None,
                ),
            )

    def update_actual_results(
        self, log_id: int, actual_pnl: float, prediction_accuracy: float
    ):
        """
        更新實際結果

        Args:
            log_id: 日誌記錄ID
            actual_pnl: 實際損益
            prediction_accuracy: 預測準確度
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE daily_analysis_logs 
                SET actual_pnl = ?, prediction_accuracy = ?
                WHERE id = ?
            """,
                (actual_pnl, prediction_accuracy, log_id),
            )

    def query_logs(
        self,
        symbol: str = None,
        date_from: str = None,
        date_to: str = None,
        event_type: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        查詢日誌記錄

        Args:
            symbol: 股票代碼
            date_from: 開始日期
            date_to: 結束日期
            event_type: 事件類型
            limit: 限制返回數量

        Returns:
            日誌記錄列表
        """
        query = """
            SELECT d.*, GROUP_CONCAT(e.event_type) as event_types
            FROM daily_analysis_logs d
            LEFT JOIN event_analysis_logs e ON d.id = e.daily_log_id
            WHERE d.session_id = ?
        """
        params = [self.session_id]

        if symbol:
            query += " AND d.symbol = ?"
            params.append(symbol)

        if date_from:
            query += " AND d.date >= ?"
            params.append(date_from)

        if date_to:
            query += " AND d.date <= ?"
            params.append(date_to)

        query += " GROUP BY d.id ORDER BY d.date DESC, d.timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            # 轉換為字典並解析JSON字段
            results = []
            for row in rows:
                record = dict(row)

                # 解析JSON字段
                for json_field in [
                    "trend_analysis",
                    "comprehensive_technical_analysis",
                    "triggered_events",
                    "llm_decision",
                    "trading_signal",
                    "strategy_state",
                ]:
                    if record[json_field]:
                        try:
                            record[json_field] = json.loads(record[json_field])
                        except json.JSONDecodeError:
                            record[json_field] = None

                results.append(record)

            return results

    def get_session_summary(self) -> Dict[str, Any]:
        """
        獲取會話摘要統計

        Returns:
            會話統計數據
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # 啟用Row工廠

            # 基本統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_days,
                    COUNT(DISTINCT symbol) as symbols_count,
                    MIN(date) as start_date,
                    MAX(date) as end_date,
                    AVG(actual_pnl) as avg_pnl,
                    SUM(actual_pnl) as total_pnl
                FROM daily_analysis_logs 
                WHERE session_id = ?
            """,
                (self.session_id,),
            )

            row = cursor.fetchone()
            basic_stats = dict(row) if row else {}

            # LLM決策統計
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_decisions,
                    AVG(CASE WHEN json_extract(llm_decision, '$.decision_made') = 1 
                        THEN 1 ELSE 0 END) as decision_rate,
                    AVG(CAST(json_extract(llm_decision, '$.confidence') AS REAL)) as avg_confidence
                FROM daily_analysis_logs 
                WHERE session_id = ? AND llm_decision IS NOT NULL
            """,
                (self.session_id,),
            )

            row = cursor.fetchone()
            llm_stats = dict(row) if row else {}

            # 事件統計
            cursor = conn.execute(
                """
                SELECT 
                    event_type,
                    COUNT(*) as count,
                    AVG(CASE WHEN severity = 'high' THEN 1 ELSE 0 END) as high_severity_rate
                FROM event_analysis_logs 
                WHERE session_id = ?
                GROUP BY event_type
                ORDER BY count DESC
            """,
                (self.session_id,),
            )

            event_stats = [dict(row) for row in cursor.fetchall()]

            return {
                "session_id": self.session_id,
                "basic_stats": basic_stats,
                "llm_stats": llm_stats,
                "event_stats": event_stats,
            }

    def export_to_json(self, filepath: str):
        """
        導出日誌到JSON文件

        Args:
            filepath: 輸出文件路徑
        """
        logs = self.query_logs(limit=None)
        summary = self.get_session_summary()

        export_data = {"session_summary": summary, "logs": logs}

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"✅ 日誌已導出到: {filepath}")
