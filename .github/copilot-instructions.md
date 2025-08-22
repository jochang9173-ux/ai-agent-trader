---
description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function **in English**.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions **in English**.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used **in English**.
- Write code with good maintainability practices, including comments on why certain design decisions were made **in English**.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments **in English**.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Language Guidelines

- **Code Comments & Documentation**: Must be written in **English** only
- **Docstrings**: Must be written in **English** only
- **Variable/Function Names**: Use **English** naming conventions
- **LLM Prompts & Instructions**: Can use **Traditional Chinese (繁體中文)**
- **Prohibited**: **Simplified Chinese (簡體中文)** is strictly forbidden throughout the entire project

## Database Configuration

### BacktestLogger Database Location
- **Correct Database Path**: `backend/data/backtest_logs.db`
- **Incorrect Path (Empty)**: `backtest_logs.db` (root directory - this file is empty)
- **Database Structure**: SQLite with tables `daily_analysis_logs` and `event_analysis_logs`
- **Key Fields**: 
  - `comprehensive_technical_analysis` (JSON): Complete technical analysis data
  - `triggered_events` (JSON): Specific trigger events for LLM decisions
  - `trend_analysis` (JSON): Trend analysis results
  - `llm_decision` (JSON): LLM decision details

### Data Format Standards
- **Boolean Values**: All boolean values in technical analysis are stored as strings ("yes"/"no") for JSON compatibility
- **Event Fields**: Use `event_type` and `severity` (not `type` and `strength`)
- **Technical Analysis**: JSON serializable with numpy type conversion via `_convert_numpy_types()`

### Database Commands
```bash
# Check data existence
sqlite3 backend/data/backtest_logs.db "SELECT COUNT(*) FROM daily_analysis_logs WHERE comprehensive_technical_analysis IS NOT NULL;"

# View latest comprehensive analysis
sqlite3 backend/data/backtest_logs.db "SELECT comprehensive_technical_analysis FROM daily_analysis_logs WHERE comprehensive_technical_analysis IS NOT NULL ORDER BY date DESC LIMIT 1;" | python -m json.tool

# Check technical events
sqlite3 backend/data/backtest_logs.db "SELECT symbol, date, json_extract(triggered_events, '$[0].event_type') as event FROM daily_analysis_logs WHERE triggered_events IS NOT NULL ORDER BY date DESC LIMIT 5;"
```

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword **in English**.
- Use blank lines to separate functions, classes, and code blocks where appropriate.
- All code comments must be written **in English**.

## Frontend Chart Components Guidelines

- **ALWAYS use SimpleTradingViewChart.tsx for price charts** - it's the only working TradingView Lightweight Charts implementation
- **NEVER use TradingChart, EnhancedLLMTradingChart, or any other chart components** - they use incorrect packages and cause runtime errors
- SimpleTradingViewChart.tsx supports: candlestick charts, volume, technical indicators (MA, RSI, BB, MACD), trading signals, and LLM decisions
- All chart-related features should be implemented through SimpleTradingViewChart.tsx parameters and props

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases **in English**.
- Write unit tests for functions and document them with docstrings explaining the test cases **in English**.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Parameters:
    radius (float): The radius of the circle.
    
    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

---

# Trading Strategy System Architecture

## API Endpoint Mapping

### Backend API Structure (FastAPI)
Base URL: `http://localhost:8000/api/v1`

#### Strategy APIs
- **Strategy List**: `GET /strategies/`
  - Returns: List of all available strategies with metadata
  
- **Strategy Parameters**: `GET /strategies/{strategy_id}/parameters`
  - Returns: Parameter specifications for a specific strategy
  - strategy_id: `moving_average_cross`, `rsi_strategy`, `bollinger_bands`, `macd_strategy`

#### Backtest APIs  
- **Run Backtest**: `POST /backtest/run`
  - Executes backtest for single strategy
  - Body: `{symbol, strategy_name, period, initial_capital, strategy_params?}`

#### Strategy Configuration APIs
- **Get All Configs**: `GET /strategy-config/strategies/configs`
- **Get Strategy Config**: `GET /strategy-config/strategies/{strategy_id}/config`
- **Validate Parameters**: `POST /strategy-config/strategies/{strategy_id}/validate`
- **Update Parameters**: `PUT /strategy-config/strategies/{strategy_id}/parameters`

#### Strategy Combination APIs
- **Create Combination**: `POST /strategy-combinations/create`
- **Backtest Combination**: `POST /strategy-combinations/backtest`

## Strategy ID Mapping System

### Unified Strategy IDs
All backend files must use these consistent strategy IDs:

```python
STRATEGY_IDS = {
    'moving_average_cross': 'MovingAverageCrossStrategy',
    'rsi_strategy': 'RSIStrategy', 
    'bollinger_bands': 'BollingerBandsStrategy',
    'macd_strategy': 'MACDStrategy'
}
```

### Files that must be updated when adding new strategies:

#### 1. Strategy Implementation Files
- **Location**: `backend/app/strategies/`
- **Files**: `{strategy_name}.py` (e.g., `bollinger_bands.py`)
- **Requirements**: 
  - Inherit from `TradingStrategy`
  - Implement `__init__(self, config: StrategyConfig)`
  - Implement `generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]`

#### 2. Strategy Registration  
- **File**: `backend/app/strategies/__init__.py`
- **Function**: `get_available_strategies()`
- **Update**: Add new strategy to returned dictionary with consistent ID

#### 3. API Configuration Files
- **File**: `backend/app/api/v1/endpoints/strategies.py`
- **Function**: `get_strategies()`
- **Update**: Add new strategy configuration to strategies list

- **File**: `backend/app/api/v1/endpoints/strategy_config.py` 
- **Variable**: `STRATEGY_ID_MAPPING`
- **Update**: Add mapping entry for new strategy

- **File**: `backend/app/api/v1/endpoints/backtests.py`
- **Variable**: `STRATEGY_ID_MAPPING` 
- **Update**: Add mapping entry for new strategy

#### 4. Strategy Parameter Specifications
- **File**: `backend/app/api/v1/endpoints/strategies.py`
- **Function**: `get_strategy_parameters()`
- **Update**: Add parameter specification for new strategy

#### 5. Strategy Combination Support
- **File**: `backend/app/api/v1/endpoints/strategy_combination.py`
- **Function**: `backtest_combination()`
- **Update**: Add strategy instantiation case for new strategy

### Frontend Type Definitions
- **File**: `frontend/src/types/index.ts`
- **Interface**: `Strategy.type` union type
- **Update**: Add new strategy type to union

## Strategy Implementation Checklist

When adding a new trading strategy, ensure all these files are updated:

### Backend Files (7 locations)
1. [ ] `backend/app/strategies/{new_strategy}.py` - Create strategy implementation
2. [ ] `backend/app/strategies/__init__.py` - Register in `get_available_strategies()`
3. [ ] `backend/app/api/v1/endpoints/strategies.py` - Add to strategy list and parameters
4. [ ] `backend/app/api/v1/endpoints/strategy_config.py` - Update `STRATEGY_ID_MAPPING`
5. [ ] `backend/app/api/v1/endpoints/backtests.py` - Update `STRATEGY_ID_MAPPING`
6. [ ] `backend/app/api/v1/endpoints/strategy_combination.py` - Add combination support
7. [ ] Update any tests in `test_*.py` files

### Frontend Files (1 location)
1. [ ] `frontend/src/types/index.ts` - Add to Strategy.type union

### Strategy Instance Creation Pattern
Always use the unified strategy instantiation function:

```python
def create_strategy_instance(strategy_class, strategy_id: str, custom_params: Dict[str, Any] = None):
    """
    Creates strategy instance with proper config handling.
    Handles both new strategies (require config) and legacy strategies.
    """
    from app.strategies import StrategyConfig
    
    default_config = StrategyConfig(
        name=strategy_id,
        description=f"{strategy_id} Strategy",
        parameters=custom_params or {}
    )
    
    try:
        return strategy_class(default_config)
    except TypeError:
        strategy_instance = strategy_class()
        if custom_params and hasattr(strategy_instance, 'update_parameters'):
            strategy_instance.update_parameters(custom_params)
        return strategy_instance
```

## Current Active Strategies

### 1. Moving Average Cross (`moving_average_cross`)
- **Class**: `MovingAverageCrossStrategy`
- **Parameters**: `short_window`, `long_window`
- **Type**: Trend-following strategy

### 2. RSI Strategy (`rsi_strategy`)
- **Class**: `RSIStrategy`  
- **Parameters**: `rsi_period`, `oversold_threshold`, `overbought_threshold`
- **Type**: Mean-reversion strategy

### 3. Bollinger Bands (`bollinger_bands`)
- **Class**: `BollingerBandsStrategy`
- **Parameters**: `period`, `std_dev`, `buy_threshold`, `sell_threshold`
- **Type**: Mean-reversion strategy

### 4. MACD Strategy (`macd_strategy`)
- **Class**: `MACDStrategy`
- **Parameters**: `fast_period`, `slow_period`, `signal_period`, `use_histogram`, `min_confidence`
- **Type**: Trend-following strategy