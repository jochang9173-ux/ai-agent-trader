"""
Enhanced Technical Analyzer for comprehensive market context analysis.
Provides detailed technical analysis beyond trigger events for LLM decision-making.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EnhancedTechnicalAnalyzer:
    """
    Comprehensive technical analysis engine that provides detailed market context
    for LLM decision-making beyond simple trigger events.
    """
    
    def __init__(self):
        """Initialize the enhanced technical analyzer."""
        self.lookback_periods = {
            'short': 5,
            'medium': 20,
            'long': 50
        }
    
    def analyze_comprehensive_context(
        self, 
        data: pd.DataFrame, 
        current_date: str,
        lookback_days: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive technical analysis context for a specific date.
        
        Args:
            data: Historical price data DataFrame
            current_date: Date to analyze (YYYY-MM-DD format)
            lookback_days: Number of days to include in analysis
            
        Returns:
            Dict containing comprehensive technical analysis
        """
        try:
            # Ensure data has proper date handling
            working_data = data.copy()
            
            # If index is date-like, reset it to create a date column
            if hasattr(working_data.index, 'date') or str(working_data.index.dtype).startswith('datetime'):
                working_data = working_data.reset_index()
                if 'index' in working_data.columns:
                    working_data.rename(columns={'index': 'date'}, inplace=True)
            
            # If we still don't have a date column, check if there's a Date column
            if 'date' not in working_data.columns and 'Date' in working_data.columns:
                working_data.rename(columns={'Date': 'date'}, inplace=True)
            
            # Convert date column to string format for comparison if it exists
            if 'date' in working_data.columns:
                working_data['date'] = pd.to_datetime(working_data['date']).dt.strftime('%Y-%m-%d')
                
                # Get data slice for analysis
                current_idx = working_data[working_data['date'] == current_date].index
                if len(current_idx) == 0:
                    logger.warning(f"No data found for date: {current_date}")
                    return self._empty_context()
            else:
                # If no date column available, use the last available data as current
                logger.warning(f"No date column found, using last data point as current")
                current_idx = [len(working_data) - 1]
            
            current_idx = current_idx[0]
            start_idx = max(0, current_idx - lookback_days)
            end_idx = min(len(working_data), current_idx + 1)
            
            analysis_data = working_data.iloc[start_idx:end_idx].copy()
            current_data = working_data.iloc[current_idx]
            
            # Comprehensive analysis
            context = {
                'date': current_date,
                'price_action': self._analyze_price_action(analysis_data, current_data),
                'moving_averages': self._analyze_moving_averages(analysis_data, current_data),
                'volume_analysis': self._analyze_volume_patterns(analysis_data, current_data),
                'volatility_analysis': self._analyze_volatility(analysis_data, current_data),
                'momentum_indicators': self._analyze_momentum(analysis_data, current_data),
                'support_resistance': self._analyze_support_resistance(analysis_data, current_data),
                'trend_analysis': self._analyze_trend_strength(analysis_data, current_data),
                'market_regime': self._classify_market_regime(analysis_data, current_data),
                'bollinger_analysis': self._analyze_bollinger_bands(analysis_data, current_data),
                'macd_analysis': self._analyze_macd(analysis_data, current_data)
            }
            
            # Convert numpy types to JSON-serializable types
            context = self._convert_numpy_types(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._empty_context()
    
    def _analyze_price_action(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze price action patterns and movements."""
        try:
            if len(data) < 2:
                return {}
            
            # Calculate price changes
            price_change = current['close'] - data['close'].iloc[-2]
            price_change_pct = (price_change / data['close'].iloc[-2]) * 100
            
            # Analyze candle patterns
            current_range = current['high'] - current['low']
            body_size = abs(current['close'] - current['open'])
            body_ratio = body_size / current_range if current_range > 0 else 0
            
            # Determine candle type
            is_bullish = current['close'] > current['open']
            upper_shadow = current['high'] - max(current['open'], current['close'])
            lower_shadow = min(current['open'], current['close']) - current['low']
            
            # Gap analysis
            gap = current['open'] - data['close'].iloc[-2] if len(data) >= 2 else 0
            gap_pct = (gap / data['close'].iloc[-2]) * 100 if len(data) >= 2 and data['close'].iloc[-2] != 0 else 0
            
            return {
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'candle_type': 'bullish' if is_bullish else 'bearish',
                'body_ratio': round(body_ratio, 3),
                'upper_shadow_ratio': round(upper_shadow / current_range, 3) if current_range > 0 else 0,
                'lower_shadow_ratio': round(lower_shadow / current_range, 3) if current_range > 0 else 0,
                'gap': round(gap, 2),
                'gap_pct': round(gap_pct, 2),
                'range': round(current_range, 2),
                'volume_to_avg_ratio': round(current['volume'] / data['volume'].mean(), 2) if data['volume'].mean() > 0 else 1
            }
        except Exception as e:
            logger.error(f"Error in price action analysis: {e}")
            return {}
    
    def _analyze_moving_averages(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze moving average relationships and trends."""
        try:
            # Calculate multiple moving averages
            ma_5 = data['close'].rolling(5).mean().iloc[-1] if len(data) >= 5 else current['close']
            ma_10 = data['close'].rolling(10).mean().iloc[-1] if len(data) >= 10 else current['close']
            ma_20 = data['close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current['close']
            
            current_price = current['close']
            
            # Calculate slopes (direction)
            ma_5_slope = self._calculate_slope(data['close'].rolling(5).mean(), 3) if len(data) >= 8 else 0
            ma_10_slope = self._calculate_slope(data['close'].rolling(10).mean(), 3) if len(data) >= 13 else 0
            ma_20_slope = self._calculate_slope(data['close'].rolling(20).mean(), 3) if len(data) >= 23 else 0
            
            # Determine MA alignment
            ma_alignment = self._get_ma_alignment(ma_5, ma_10, ma_20)
            
            return {
                'ma_5': round(ma_5, 2),
                'ma_10': round(ma_10, 2),
                'ma_20': round(ma_20, 2),
                'price_vs_ma_5': round(((current_price / ma_5) - 1) * 100, 2) if ma_5 != 0 else 0,
                'price_vs_ma_10': round(((current_price / ma_10) - 1) * 100, 2) if ma_10 != 0 else 0,
                'price_vs_ma_20': round(((current_price / ma_20) - 1) * 100, 2) if ma_20 != 0 else 0,
                'ma_5_slope': round(ma_5_slope, 4),
                'ma_10_slope': round(ma_10_slope, 4),
                'ma_20_slope': round(ma_20_slope, 4),
                'ma_alignment': ma_alignment,
                'above_all_mas': "yes" if (current_price > ma_5 and current_price > ma_10 and current_price > ma_20) else "no",
                'below_all_mas': "yes" if (current_price < ma_5 and current_price < ma_10 and current_price < ma_20) else "no"
            }
        except Exception as e:
            logger.error(f"Error in moving average analysis: {e}")
            return {}
    
    def _analyze_volume_patterns(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze volume patterns and anomalies."""
        try:
            current_volume = current['volume']
            avg_volume = data['volume'].mean()
            volume_std = data['volume'].std()
            
            # Volume ratios
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_z_score = (current_volume - avg_volume) / volume_std if volume_std > 0 else 0
            
            # Volume trend
            volume_trend = self._calculate_slope(data['volume'], 3) if len(data) >= 4 else 0
            
            # Price-volume relationship
            price_change = current['close'] - data['close'].iloc[-2] if len(data) >= 2 else 0
            price_volume_divergence = self._detect_price_volume_divergence(data)
            
            return {
                'current_volume': int(current_volume),
                'avg_volume': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'volume_z_score': round(volume_z_score, 2),
                'volume_trend': round(volume_trend, 2),
                'is_high_volume': "yes" if (volume_ratio > 1.5) else "no",
                'is_very_high_volume': "yes" if (volume_ratio > 2.0) else "no",
                'is_low_volume': "yes" if (volume_ratio < 0.5) else "no",
                'price_volume_divergence': "yes" if price_volume_divergence else "no",
                'volume_confirmation': "yes" if ((price_change > 0 and volume_ratio > 1) or (price_change < 0 and volume_ratio > 1)) else "no"
            }
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {}
    
    def _analyze_volatility(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze volatility patterns and changes."""
        try:
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(data)
            current_tr = max(
                current['high'] - current['low'],
                abs(current['high'] - data['close'].iloc[-2]) if len(data) >= 2 else 0,
                abs(current['low'] - data['close'].iloc[-2]) if len(data) >= 2 else 0
            )
            
            # Price volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0  # Annualized
            
            # Volatility percentile
            historical_vol = data['close'].pct_change().rolling(20).std() if len(data) >= 20 else pd.Series([volatility])
            vol_percentile = (volatility > historical_vol).mean() * 100 if len(historical_vol) > 0 else 50
            
            return {
                'atr': round(atr, 2),
                'current_true_range': round(current_tr, 2),
                'volatility_annualized': round(volatility * 100, 2),
                'volatility_percentile': round(vol_percentile, 1),
                'is_high_volatility': "yes" if (vol_percentile > 80) else "no",
                'is_low_volatility': "yes" if (vol_percentile < 20) else "no",
                'atr_vs_price': round((atr / current['close']) * 100, 2) if current['close'] != 0 else 0
            }
        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return {}
    
    def _analyze_momentum(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze momentum indicators."""
        try:
            # RSI calculation
            rsi = self._calculate_rsi(data['close'])
            
            # Rate of Change
            roc_5 = ((current['close'] / data['close'].iloc[-6]) - 1) * 100 if len(data) >= 6 else 0
            roc_10 = ((current['close'] / data['close'].iloc[-11]) - 1) * 100 if len(data) >= 11 else 0
            
            # Momentum classification
            momentum_strength = self._classify_momentum_strength(rsi, roc_5, roc_10)
            
            return {
                'rsi': round(rsi, 2),
                'rsi_condition': self._classify_rsi(rsi),
                'roc_5_day': round(roc_5, 2),
                'roc_10_day': round(roc_10, 2),
                'momentum_strength': momentum_strength,
                'is_oversold': "yes" if (rsi < 30) else "no",
                'is_overbought': "yes" if (rsi > 70) else "no",
                'is_neutral': "yes" if (30 <= rsi <= 70) else "no"
            }
        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return {}
    
    def _analyze_support_resistance(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze support and resistance levels."""
        try:
            # Find recent highs and lows
            recent_highs = data['high'].nlargest(3).values
            recent_lows = data['low'].nsmallest(3).values
            
            current_price = current['close']
            
            # Distance to key levels
            resistance_distance = min([(h - current_price) for h in recent_highs if h > current_price], default=0)
            support_distance = min([(current_price - l) for l in recent_lows if l < current_price], default=0)
            
            # Identify if near key levels
            near_resistance = resistance_distance < current_price * 0.02  # Within 2%
            near_support = support_distance < current_price * 0.02  # Within 2%
            
            return {
                'nearest_resistance': round(min([h for h in recent_highs if h > current_price], default=current_price), 2),
                'nearest_support': round(max([l for l in recent_lows if l < current_price], default=current_price), 2),
                'resistance_distance_pct': round((resistance_distance / current_price) * 100, 2) if current_price != 0 else 0,
                'support_distance_pct': round((support_distance / current_price) * 100, 2) if current_price != 0 else 0,
                'near_resistance': "yes" if near_resistance else "no",
                'near_support': "yes" if near_support else "no",
                'between_levels': "yes" if (not near_resistance and not near_support) else "no"
            }
        except Exception as e:
            logger.error(f"Error in support/resistance analysis: {e}")
            return {}
    
    def _analyze_trend_strength(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze trend strength and direction."""
        try:
            # Calculate trend using linear regression
            if len(data) < 5:
                return {'trend_direction': 'neutral', 'trend_strength': 0}
            
            x = np.arange(len(data))
            y = data['close'].values
            
            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            
            # Classify trend
            trend_direction = 'bullish' if slope > 0 else 'bearish' if slope < 0 else 'neutral'
            trend_strength = r_squared  # R-squared as strength measure
            
            # ADX-like calculation (simplified)
            adx_value = self._calculate_simple_adx(data)
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 3),
                'slope': round(slope, 4),
                'r_squared': round(r_squared, 3),
                'adx_value': round(adx_value, 2),
                'strong_trend': "yes" if (trend_strength > 0.7 and adx_value > 25) else "no",
                'weak_trend': "yes" if (trend_strength < 0.3 or adx_value < 20) else "no"
            }
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {}
    
    def _classify_market_regime(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Classify current market regime."""
        try:
            # Get trend and volatility info
            trend_info = self._analyze_trend_strength(data, current)
            volatility_info = self._analyze_volatility(data, current)
            
            # Classify regime
            if trend_info.get('strong_trend', False):
                if volatility_info.get('is_high_volatility', False):
                    regime = 'trending_volatile'
                else:
                    regime = 'trending_stable'
            elif volatility_info.get('is_high_volatility', False):
                regime = 'sideways_volatile'
            else:
                regime = 'sideways_stable'
            
            return {
                'market_regime': regime,
                'regime_description': self._get_regime_description(regime),
                'is_trending': trend_info.get('strong_trend', False),
                'is_volatile': volatility_info.get('is_high_volatility', False)
            }
        except Exception as e:
            logger.error(f"Error in market regime classification: {e}")
            return {}
    
    def _analyze_bollinger_bands(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze Bollinger Bands patterns."""
        try:
            if len(data) < 20:
                return {}
            
            # Calculate Bollinger Bands
            bb_period = min(20, len(data))
            bb_middle = data['close'].rolling(bb_period).mean().iloc[-1]
            bb_std = data['close'].rolling(bb_period).std().iloc[-1]
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            
            current_price = current['close']
            
            # Position relative to bands
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            
            # Band width (volatility measure)
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle != 0 else 0
            
            # Band squeeze detection
            bb_width_avg = data['close'].rolling(bb_period).std().rolling(10).mean().iloc[-1] if len(data) >= 30 else bb_std
            is_squeeze = bb_std < bb_width_avg * 0.8 if bb_width_avg != 0 else False
            
            return {
                'bb_upper': round(bb_upper, 2),
                'bb_middle': round(bb_middle, 2),
                'bb_lower': round(bb_lower, 2),
                'bb_position': round(bb_position, 3),
                'bb_width': round(bb_width, 2),
                'above_upper_band': "yes" if (current_price > bb_upper) else "no",
                'below_lower_band': "yes" if (current_price < bb_lower) else "no",
                'near_upper_band': "yes" if (bb_position > 0.8) else "no",
                'near_lower_band': "yes" if (bb_position < 0.2) else "no",
                'is_squeeze': "yes" if is_squeeze else "no",
                'potential_breakout': "yes" if (is_squeeze and bb_position > 0.8 or bb_position < 0.2) else "no"
            }
        except Exception as e:
            logger.error(f"Error in Bollinger Bands analysis: {e}")
            return {}
    
    def _analyze_macd(self, data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
        """Analyze MACD patterns."""
        try:
            if len(data) < 26:
                return {}
            
            # Calculate MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            # MACD analysis
            macd_cross = self._detect_macd_cross(macd_line, signal_line)
            histogram_trend = self._calculate_slope(histogram, 3) if len(histogram) >= 4 else 0
            
            return {
                'macd_line': round(current_macd, 4),
                'signal_line': round(current_signal, 4),
                'histogram': round(current_histogram, 4),
                'macd_above_signal': "yes" if (current_macd > current_signal) else "no",
                'macd_cross': macd_cross,
                'histogram_trend': round(histogram_trend, 4),
                'divergence_potential': "yes" if (abs(histogram_trend) > 0.001) else "no",
                'macd_position': 'bullish' if current_macd > current_signal else 'bearish'
            }
        except Exception as e:
            logger.error(f"Error in MACD analysis: {e}")
            return {}
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        else:
            return obj
    
    # Helper methods
    def _empty_context(self) -> Dict[str, Any]:
        """Return empty context structure with default values."""
        return {
            'date': None,
            'price_action': {
                'price_change_pct': 0.0,
                'candle_type': 'neutral',
                'volatility_level': 'normal'
            },
            'moving_averages': {
                'ma5_trend': 'neutral',
                'ma20_trend': 'neutral',
                'ma_alignment': 'mixed'
            },
            'volume_analysis': {
                'volume_trend': 'normal',
                'volume_strength': 'average'
            },
            'volatility_analysis': {
                'current_volatility': 'normal',
                'volatility_trend': 'stable'
            },
            'momentum_indicators': {
                'rsi_signal': 'neutral',
                'momentum_strength': 'weak'
            },
            'support_resistance': {
                'support_level': None,
                'resistance_level': None,
                'position_relative': 'middle'
            },
            'trend_analysis': {
                'short_term': 'neutral',
                'medium_term': 'neutral',
                'long_term': 'neutral'
            },
            'market_regime': {
                'regime': 'consolidation',
                'confidence': 0.5
            },
            'bollinger_analysis': {
                'position': 'middle',
                'squeeze': "no",
                'expansion': "no"
            },
            'macd_analysis': {
                'signal': 'neutral',
                'histogram_trend': 'flat',
                'divergence': None
            }
        }
    
    def _calculate_slope(self, series: pd.Series, periods: int) -> float:
        """Calculate slope of a series over specified periods."""
        try:
            if len(series) < periods + 1:
                return 0
            recent_data = series.tail(periods + 1).dropna()
            if len(recent_data) < 2:
                return 0
            x = np.arange(len(recent_data))
            slope, _ = np.polyfit(x, recent_data.values, 1)
            return slope
        except:
            return 0
    
    def _get_ma_alignment(self, ma_5: float, ma_10: float, ma_20: float) -> str:
        """Determine moving average alignment."""
        if ma_5 > ma_10 > ma_20:
            return 'bullish_alignment'
        elif ma_5 < ma_10 < ma_20:
            return 'bearish_alignment'
        else:
            return 'mixed_alignment'
    
    def _detect_price_volume_divergence(self, data: pd.DataFrame) -> bool:
        """Detect price-volume divergence."""
        try:
            if len(data) < 4:
                return False
            
            price_trend = self._calculate_slope(data['close'], 3)
            volume_trend = self._calculate_slope(data['volume'], 3)
            
            # Divergence if trends are opposite
            return (price_trend > 0 and volume_trend < 0) or (price_trend < 0 and volume_trend > 0)
        except:
            return False
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            if len(data) < 2:
                return 0
            
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(min(period, len(data))).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0
        except:
            return 0
    
    def _calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        try:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=min(period, len(delta))).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=min(period, len(delta))).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _classify_rsi(self, rsi: float) -> str:
        """Classify RSI condition."""
        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        else:
            return 'neutral'
    
    def _classify_momentum_strength(self, rsi: float, roc_5: float, roc_10: float) -> str:
        """Classify momentum strength."""
        if rsi > 70 and roc_5 > 5:
            return 'strong_bullish'
        elif rsi < 30 and roc_5 < -5:
            return 'strong_bearish'
        elif rsi > 50 and roc_5 > 0:
            return 'moderate_bullish'
        elif rsi < 50 and roc_5 < 0:
            return 'moderate_bearish'
        else:
            return 'neutral'
    
    def _calculate_simple_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate simplified ADX."""
        try:
            if len(data) < period:
                return 0
            
            # Simplified directional movement
            dm_plus = (data['high'] - data['high'].shift(1)).where(
                (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']), 0
            )
            dm_minus = (data['low'].shift(1) - data['low']).where(
                (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)), 0
            )
            
            tr = pd.concat([
                data['high'] - data['low'],
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            ], axis=1).max(axis=1)
            
            di_plus = 100 * (dm_plus.rolling(period).sum() / tr.rolling(period).sum())
            di_minus = 100 * (dm_minus.rolling(period).sum() / tr.rolling(period).sum())
            
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(period).mean().iloc[-1]
            
            return adx if not pd.isna(adx) else 0
        except:
            return 0
    
    def _get_regime_description(self, regime: str) -> str:
        """Get description for market regime."""
        descriptions = {
            'trending_volatile': 'Strong trend with high volatility',
            'trending_stable': 'Clear trend with low volatility',
            'sideways_volatile': 'Choppy market with high volatility',
            'sideways_stable': 'Range-bound market with low volatility'
        }
        return descriptions.get(regime, 'Unknown regime')
    
    def _detect_macd_cross(self, macd_line: pd.Series, signal_line: pd.Series) -> str:
        """Detect MACD signal line crosses."""
        try:
            if len(macd_line) < 2:
                return 'none'
            
            current_above = macd_line.iloc[-1] > signal_line.iloc[-1]
            previous_above = macd_line.iloc[-2] > signal_line.iloc[-2]
            
            if current_above and not previous_above:
                return 'bullish_cross'
            elif not current_above and previous_above:
                return 'bearish_cross'
            else:
                return 'none'
        except:
            return 'none'
