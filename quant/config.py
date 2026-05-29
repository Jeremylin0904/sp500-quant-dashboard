from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep old locked files under quant/data/ if present, but new pipeline uses data_raw/
DATA_RAW_DIR = ROOT / "data_raw"
DATA_CURATED_DIR = ROOT / "data_curated"
FEATURES_DIR = ROOT / "features"
LABELS_DIR = ROOT / "labels"
MODEL_DIR = ROOT / "model"
BACKTEST_DIR = ROOT / "backtest"

# MVP file names (expected under data_raw/)
PRICES_DAILY_CSV = DATA_RAW_DIR / "prices_daily.csv"
CONSTITUENTS_CSV = DATA_RAW_DIR / "constituents.csv"
INCOME_Q_CSV = DATA_RAW_DIR / "income_quarterly.csv"
BALANCE_Q_CSV = DATA_RAW_DIR / "balance_quarterly.csv"
CASHFLOW_Q_CSV = DATA_RAW_DIR / "cashflow_quarterly.csv"  # optional

# S&P 500 universe + benchmark (SPY ≈ cap-weighted S&P 500)
DATA_DIR = ROOT / "data"
SP500_CONSTITUENTS_PATH = DATA_DIR / "sp500_constituents.csv"
BENCHMARK_SYMBOL = "SPY"
BENCHMARK_ID = "spy_mcap_weighted"

# Label: top N by excess return vs SPY (binary y_t=1)
TOP_N_LABEL = 30
VOL_FLOOR_DAILY = 0.001  # floor for inverse-vol portfolio weights

# Portfolio: model picks top K (aligned with label Top30)
TOP_N_HOLDINGS = 30

# Portfolio weighting: scheme (+ T for softmax variants) chosen via CV OOF with model pool
WEIGHT_SCHEME = "inv_vol"  # fallback if meta missing
SOFTMAX_TEMPERATURE_POOL = [0.1, 0.2]  # legacy alias; see WEIGHT_SCHEME_POOL
PORTFOLIO_HP_SELECT_METRIC = "ann_sharpe_excess"  # CV OOF metric for (model, weight) grid

# Each entry: weight_scheme + optional softmax_temperature (None = not used)
WEIGHT_SCHEME_POOL: list[dict] = [
    {"weight_scheme": "inv_vol"},
    {"weight_scheme": "equal"},
    {"weight_scheme": "rank_linear"},
    {"weight_scheme": "score_inv_vol"},
    {"weight_scheme": "softmax", "softmax_temperature": 0.1},
    {"weight_scheme": "softmax", "softmax_temperature": 0.2},
    {"weight_scheme": "softmax_sharpe", "softmax_temperature": 0.1},
    {"weight_scheme": "softmax_sharpe", "softmax_temperature": 0.2},
]

# Modeling
TARGET_COL = "y_next"
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42
# Weighted BCE: sample_weight 1 for negatives, n_neg/n_pos for positives (per train fold)
BCE_POS_WEIGHT_MODE = "balanced"  # balanced | none
# TTM needs 4 reported quarters; Q1–Q4 (first FEATURE_WARMUP_QUARTERS) excluded from train/pool search
FEATURE_WARMUP_QUARTERS = 4
# Expanding walk-forward: 8 valid quarters after warmup before first CV OOF predict
CV_MIN_TRAIN_QUARTERS = 8
CV_TIME_BUDGET_PER_FOLD = 30  # legacy; CV OOF now uses fixed configs from pool
AUTOML_TIME_BUDGET_CV_SELECT = 60  # legacy alias for pool search budget
AUTOML_TIME_BUDGET_POOL_SEARCH = 120  # AutoML on post-warmup .. pre-OOS quarters → hyperparameter pool
HP_POOL_TOP_K = 5  # top-K unique configs from pool search → 8-fold OOF average → pick best
AUTOML_TIME_BUDGET_FINAL = 60
WF_OOS_QUARTERS = 4  # final OOS: last N quarters (expanding retrain, frozen config)
WF_RECENT_QUARTERS = WF_OOS_QUARTERS  # legacy alias

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"

# --- Data paths ---
PRICES_PATH = DATA_DIR / "sp500_prices.csv"
FUNDAMENTALS_PATH = DATA_DIR / "simfin" / "sp500_fundamentals_quarterly.csv"
CONSTITUENTS_PATH = DATA_DIR / "sp500_constituents.csv"
FF5_PATH = DATA_DIR / "french" / "french_ff5_monthly.csv"
MOM_PATH = DATA_DIR / "french" / "french_momentum_monthly.csv"

# --- Feature engineering settings ---
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99
USE_TTM = True
GROWTH_DENOM_THRESHOLD = 1e-6

# --- Modeling settings ---
TARGET_COL = "y_next"
TRAIN_VAL_SPLIT = 0.8
AUTOML_TIME_BUDGET = 120
REJECT_LOOKAHEAD = True
MAX_MISSING_RATIO = 0.3

# Feature columns stored in feature mart
FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy",
    "operating_margin", "net_margin", "roe", "roa",
    "pe_ratio", "pb_ratio", "ev_to_ebit", "ebit_to_tev", "fcf_yield",
    "debt_to_equity",
    "sector_rank_revenue_growth", "sector_rank_roe", "sector_rank_ebit_to_tev",
    "sector_rank_fcf_yield", "sector_rank_debt_to_equity", "sector_rank_operating_margin",
    "return_on_equity", "net_profit_margin", "profitability_to_book_proxy", "eps_diluted",
]

# Feature columns used for AutoML (excludes alias duplicates)
MODEL_FEATURE_COLS = [
    "revenue_growth_yoy", "eps_growth_yoy",
    "operating_margin", "net_margin", "roe", "roa",
    "pe_ratio", "pb_ratio", "ev_to_ebit", "ebit_to_tev", "fcf_yield",
    "debt_to_equity",
    "sector_rank_revenue_growth", "sector_rank_roe", "sector_rank_ebit_to_tev",
    "sector_rank_fcf_yield", "sector_rank_debt_to_equity", "sector_rank_operating_margin",
    "eps_diluted",
]
