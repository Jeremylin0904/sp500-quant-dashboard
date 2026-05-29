from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import backtest, chat, factor, holdings, model
from backend.services.data_service import clear_cache

app = FastAPI(title="Interview MVP API", version="2.1.0")


@app.on_event("startup")
def _refresh_data_cache() -> None:
    clear_cache()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(backtest.router)
app.include_router(holdings.router)
app.include_router(model.router)
app.include_router(factor.router)
app.include_router(chat.router)


@app.get("/api/health")
def health():
    from quant.config import BACKTEST_DIR
    from backend.services.data_service import clear_cache, get_holdings_by_signal_quarter

    clear_cache()
    q_path = BACKTEST_DIR / "holdings_by_signal_quarter.json"
    h = get_holdings_by_signal_quarter()
    return {
        "status": "ok",
        "version": "2.1.0",
        "holdings_signal_quarter_file": str(q_path),
        "holdings_signal_quarter_exists": q_path.exists(),
        "n_signal_quarters": len(h),
    }

