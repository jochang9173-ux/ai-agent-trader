"""
Stock data fetcher using yfinance utilities.
Provides enhanced stock data fetching with proper error handling and analysis periods.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

from .yfinance_utils import (
    DataRetrievalError,
    EmptyDataError,
    TickerNotFoundError,
    YFinanceError,
    YFinanceService,
)

# Setup logging
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Enhanced stock data fetcher with analysis period support.

    This class provides stock data fetching capabilities with support for
    different analysis periods and technical indicator calculation requirements.
    """

    # Analysis period mapping (what user wants to analyze)
    ANALYSIS_PERIOD_DAYS = {
        "1mo": 30,
        "2mo": 60,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
    }

    # Data fetch period mapping (more data needed for technical indicators)
    # We need extra data for technical indicators calculation
    DATA_FETCH_DAYS = {
        "1mo": 120,  # Fetch 4 months to analyze 1 month
        "2mo": 150,  # Fetch 5 months to analyze 2 months
        "3mo": 180,  # Fetch 6 months to analyze 3 months
        "6mo": 365,  # Fetch 1 year to analyze 6 months
        "1y": 730,  # Fetch 2 years to analyze 1 year
        "2y": 1095,  # Fetch 3 years to analyze 2 years
    }

    @classmethod
    def fetch_stock_data(cls, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for the given symbol and period with analysis metadata.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL', '2330.TW')
            period (str): Period string (e.g., '1mo', '3mo', '6mo', '1y', '2y')

        Returns:
            Optional[pd.DataFrame]: DataFrame with stock data and analysis metadata,
                                  or None if failed

        Raises:
            None: All exceptions are caught and logged, returns None on failure
        """
        try:
            # Validate inputs
            if not symbol or not isinstance(symbol, str):
                logger.error("Invalid symbol provided")
                return None

            if period not in cls.ANALYSIS_PERIOD_DAYS:
                logger.warning(f"Unknown period '{period}', using default '3mo'")
                period = "3mo"

            # Calculate date ranges
            end_date = datetime.now()
            analysis_days = cls.ANALYSIS_PERIOD_DAYS.get(period, 90)
            fetch_days = cls.DATA_FETCH_DAYS.get(period, 180)

            analysis_start_date = end_date - timedelta(days=analysis_days)
            data_fetch_start_date = end_date - timedelta(days=fetch_days)

            # Format dates
            start_date_str = data_fetch_start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            analysis_start_str = analysis_start_date.strftime("%Y-%m-%d")

            logger.info(
                f"Fetching data for {symbol} from {start_date_str} to {end_date_str}"
            )

            # Fetch data using YFinanceService
            df = YFinanceService.get_stock_data(
                symbol=symbol, start_date=start_date_str, end_date=end_date_str
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Add metadata about analysis period to the DataFrame
            df.attrs.update(
                {
                    "analysis_start_date": analysis_start_str,
                    "analysis_period": period,
                    "total_fetch_days": fetch_days,
                    "analysis_days": analysis_days,
                    "symbol": symbol,
                    "fetch_start_date": start_date_str,
                    "fetch_end_date": end_date_str,
                }
            )

            logger.info(f"Successfully fetched {len(df)} rows of data for {symbol}")
            return df

        except TickerNotFoundError as e:
            logger.error(f"Ticker not found for {symbol}: {e}")
            return None

        except (DataRetrievalError, EmptyDataError) as e:
            logger.error(f"Data retrieval error for {symbol}: {e}")
            return None

        except YFinanceError as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error fetching data for {symbol}: {e}")
            return None

    @classmethod
    def get_analysis_period_info(cls, period: str) -> Dict[str, int]:
        """
        Get information about analysis and fetch periods.

        Args:
            period (str): Period string

        Returns:
            Dict[str, int]: Dictionary containing period information
        """
        return {
            "analysis_days": cls.ANALYSIS_PERIOD_DAYS.get(period, 90),
            "fetch_days": cls.DATA_FETCH_DAYS.get(period, 180),
            "extra_days": cls.DATA_FETCH_DAYS.get(period, 180)
            - cls.ANALYSIS_PERIOD_DAYS.get(period, 90),
        }

    @classmethod
    def get_supported_periods(cls) -> list:
        """
        Get list of supported analysis periods.

        Returns:
            list: List of supported period strings
        """
        return list(cls.ANALYSIS_PERIOD_DAYS.keys())


def fetch_stock_data(symbol: str, period: str) -> Optional[pd.DataFrame]:
    """
    Convenience function for fetching stock data.

    Args:
        symbol (str): Stock ticker symbol
        period (str): Period string (e.g., '3mo', '6mo', '1y')

    Returns:
        Optional[pd.DataFrame]: DataFrame with stock data or None if failed
    """
    return StockDataFetcher.fetch_stock_data(symbol, period)
