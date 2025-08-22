"""
Technical indicator calculation and stock analysis logic.
All functions are pure and stateless, suitable for use in async API services.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def calculate_moving_averages(
    df: pd.DataFrame, short_window: int = 10, long_window: int = 20
) -> pd.DataFrame:
    """Calculate moving averages for a DataFrame"""
    df = df.copy()
    df["ma_10"] = df["close"].rolling(window=short_window).mean()
    df["ma_20"] = df["close"].rolling(window=long_window).mean()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    return df


def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate RSI for a DataFrame"""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def calculate_rsi_series(close_prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI for a Series (used in stocks.py)"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bollinger_bands(
    df: pd.DataFrame, window: int = 20, num_std_dev: int = 2
) -> pd.DataFrame:
    """Calculate Bollinger Bands for a DataFrame"""
    df = df.copy()

    # Calculate moving average and standard deviation, handle NaN values
    close_series = df["close"].ffill().bfill()
    df["bb_middle"] = close_series.rolling(window=window, min_periods=1).mean()
    rolling_std = close_series.rolling(window=window, min_periods=1).std().fillna(0)

    df["bb_upper"] = df["bb_middle"] + num_std_dev * rolling_std
    df["bb_lower"] = df["bb_middle"] - num_std_dev * rolling_std

    # Fill possible NaN values
    df["bb_middle"] = df["bb_middle"].fillna(df["close"])
    df["bb_upper"] = df["bb_upper"].fillna(df["close"])
    df["bb_lower"] = df["bb_lower"].fillna(df["close"])

    return df


def calculate_bollinger_bands_series(
    close_prices: pd.Series, window: int = 20, num_std_dev: int = 2
) -> Optional[Dict[str, pd.Series]]:
    """Calculate Bollinger Bands for a Series (used in stocks.py)"""
    try:
        if len(close_prices) < window:
            # If insufficient data, use available data
            min_periods = max(1, len(close_prices) // 2)
        else:
            min_periods = window

        middle = close_prices.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = close_prices.rolling(window=window, min_periods=min_periods).std()
        upper = middle + num_std_dev * rolling_std
        lower = middle - num_std_dev * rolling_std

        return {"upper": upper, "lower": lower, "middle": middle}
    except Exception:
        return None


def calculate_macd(
    df: pd.DataFrame,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Calculate MACD for a DataFrame"""
    df = df.copy()
    df["ema_short"] = df["close"].ewm(span=short_period, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=long_period, adjust=False).mean()
    df["macd"] = df["ema_short"] - df["ema_long"]
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


def calculate_macd_series(
    close_prices: pd.Series,
    short_period: int = 12,
    long_period: int = 26,
    signal_period: int = 9,
) -> Optional[Dict[str, pd.Series]]:
    """Calculate MACD for a Series (used in stocks.py)"""
    try:
        ema_short = close_prices.ewm(span=short_period, adjust=False).mean()
        ema_long = close_prices.ewm(span=long_period, adjust=False).mean()
        macd = ema_short - ema_long
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal_line

        return {"macd": macd, "signal": signal_line, "histogram": histogram}
    except Exception:
        return None


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate ATR for a DataFrame"""
    df = df.copy()
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = abs(df["high"] - df["close"].shift())
    df["low_close"] = abs(df["low"] - df["close"].shift())
    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=window).mean()
    return df


def calculate_vma(
    df: pd.DataFrame, short_window: int = 5, long_window: int = 20
) -> pd.DataFrame:
    df["vma_short"] = df["volume"].rolling(window=short_window).mean()
    df["vma_long"] = df["volume"].rolling(window=long_window).mean()
    return df


def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    moving_avg = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(
        lambda x: abs(x - x.mean()).mean(), raw=True
    )
    df["cci"] = (typical_price - moving_avg) / (0.015 * mean_deviation)
    return df


def calculate_kdj(df: pd.DataFrame, window: int = 9) -> pd.DataFrame:
    low_min = df["low"].rolling(window=window).min()
    high_max = df["high"].rolling(window=window).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    df["kdj_k"] = rsv.ewm(com=2).mean()
    df["kdj_d"] = df["kdj_k"].ewm(com=2).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    return df


def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df["up_move"] = df["high"].diff()
    df["down_move"] = df["low"].diff()
    df["down_move"] = df["down_move"].apply(lambda x: abs(x) if pd.notnull(x) else x)
    df["plus_dm"] = ((df["up_move"] > df["down_move"]) & (df["up_move"] > 0)) * df[
        "up_move"
    ]
    df["minus_dm"] = ((df["down_move"] > df["up_move"]) & (df["down_move"] > 0)) * df[
        "down_move"
    ]
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (df["plus_dm"].rolling(window=window).sum() / atr)
    minus_di = 100 * (df["minus_dm"].rolling(window=window).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    df["adx"] = dx.rolling(window=window).mean()
    return df
