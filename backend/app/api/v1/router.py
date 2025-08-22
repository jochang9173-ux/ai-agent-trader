"""
API v1 router configuration.
Includes all endpoint routers for version 1 of the API.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import llm_stream, backtest_analysis, daily_feedback

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(llm_stream.router, prefix="/llm-stream", tags=["llm-stream"])
api_router.include_router(backtest_analysis.router, prefix="/backtest", tags=["backtest-analysis"])
api_router.include_router(daily_feedback.router, prefix="/daily", tags=["daily-feedback"])
