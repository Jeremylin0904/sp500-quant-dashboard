from __future__ import annotations

from fastapi import APIRouter

from backend.services.data_service import get_factor_analysis

router = APIRouter(prefix="/api/factor", tags=["factor"])


@router.get("/analysis")
def analysis():
    """Fama-French 5-factor + Momentum regression results (CV OOF / Final OOS / all)."""
    return get_factor_analysis()
