import { useEffect, useMemo, useState } from "react";
import { PerformanceChart } from "./components/PerformanceChart";
import { ScatterChart } from "./components/ScatterChart";
import { MethodologyView } from "./components/MethodologyView";
import { VariablesView, type ModelVariables } from "./components/VariablesView";
import "./App.css";

type MonthlyRow = {
  month: string;
  signal_quarter?: string;
  portfolio_return: number;
  benchmark_return: number;
  excess_return: number;
  portfolio_vol: number;
  strategy_nav: number;
  benchmark_nav: number;
};

type SelectedH = {
  symbol: string;
  weight: number;
  score: number;
  daily_vol: number;
  realized_return?: number | null;
  realized_excess?: number | null;
  is_top30_actual?: boolean;
};
type ActualH = { symbol: string; excess_return: number; return: number; y_label: number };
type HoldingsQuarter = {
  selected: SelectedH[];
  actual_top30: ActualH[];
  signal_quarter?: string;
  realized_quarter?: string;
  hit_count?: number;
  hit_rate?: number;
  label_note?: string;
};

type DailyRow = {
  date: string;
  strategy_nav: number;
  benchmark_nav: number;
  daily_return: number;
};

type DDHolding = { symbol: string; weight: number; daily_ret: number; contrib: number };
type DDBreakdown = {
  date: string;
  portfolio_return: number;
  n_held: number;
  holdings: DDHolding[];
};
type DrawdownBreakdowns = {
  cv_oof?: DDBreakdown[];
  final_oos?: DDBreakdown[];
  merged?: DDBreakdown[];
};

type PerfSlice = Record<string, number | string>;

type PerfReport = {
  benchmark?: string;
  benchmark_symbol?: string;
  top_n_holdings?: number;
  label_top_n?: number;
  weighting?: string;
  in_sample?: PerfSlice;
  out_of_sample?: PerfSlice;
};

type Cm = {
  tn?: number;
  fp?: number;
  fn?: number;
  tp?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  specificity?: number;
  f1?: number;
  n?: number;
};

type ModelSummary = {
  model?: string;
  best_estimator?: string;
  loss?: { name?: string };
  cv?: { oof_roc_auc?: number; oof_log_loss?: number };
  cv_eval?: {
    spearman_mean?: number;
    top30_precision?: number;
    top30_recall?: number;
    confusion_matrix?: Cm;
    per_quarter?: Array<{
      quarter: string;
      spearman: number;
      "precision@30"?: number;
      "recall@30"?: number;
    }>;
  };
  holdout?: { val_roc_auc?: number; val_log_loss?: number };
  hp_pool?: {
    top_k?: number;
    selected_pool_rank?: number;
    selected_estimator?: string;
    selected_weight_scheme?: string;
    weight_spec_label?: string;
    portfolio_select_metric?: string;
    evaluations?: Array<{
      pool_rank: number;
      estimator: string;
      weight_scheme?: string;
      weight_spec_label?: string;
      softmax_temperature?: number | null;
      mean_fold_log_loss?: number;
      oof_portfolio_ann_sharpe_excess?: number;
      pooled_oof_roc_auc?: number;
      selected?: boolean;
    }>;
  };
  oos_eval?: {
    spearman_mean?: number;
    top30_precision?: number;
    top30_recall?: number;
    confusion_matrix?: Cm;
    per_quarter?: Array<{
      quarter: string;
      spearman: number;
      precision?: number;
      recall?: number;
      "precision@30"?: number;
      "recall@30"?: number;
    }>;
  };
  portfolio?: { benchmark_symbol?: string; top_n_holdings?: number };
  universe?: {
    n_companies?: number;
    n_sp500_constituents?: number;
    coverage_note?: string;
    per_quarter?: Record<string, number>;
  };
  all_quarters?: string[];
  final_oos?: { roc_auc?: number };
  walk_forward?: {
    feature_warmup_quarters?: number;
    feature_warmup_excluded?: string[];
    selection_train_quarters?: string[];
    pool_search_time_budget_sec?: number;
    hp_pool_top_k?: number;
    n_cv_folds?: number;
    n_oos_folds?: number;
    min_train_quarters?: number;
    folds?: Array<{
      fold: number;
      phase: string;
      n_train_quarters?: number;
      train_start?: string;
      train_end?: string;
      pred_quarter?: string;
      pred_roc_auc?: number;
    }>;
    cv_oof_quarters?: string[];
    final_oos_quarters?: string[];
  };
};

type FactorRow = { factor: string; beta: number; t_stat: number; p_value: number };
type FactorSeg = {
  segment_label?: string;
  n_months?: number;
  months?: string[];
  alpha_monthly?: number;
  alpha_annualized_compound?: number;
  alpha_t_stat?: number;
  alpha_p_value?: number;
  r_squared?: number;
  adj_r_squared?: number;
  factors?: FactorRow[];
};
type FactorAnalysis = {
  model?: string;
  source?: string;
  regression_spec?: string;
  factor_definitions?: Record<string, string>;
  factor_data_window?: { start?: string; end?: string };
  segments?: Record<string, FactorSeg | null>;
};

// In dev, Vite proxies /api -> FastAPI (localhost:8001). In the production
// build the app is fully static (e.g. GitHub Pages): API responses are
// pre-rendered to JSON files under <base>/api/<path>.json by
// scripts/export_static_api.py.
declare const __BUILD_ID__: string;

function apiUrl(path: string): string {
  if (!import.meta.env.PROD) return path;
  const clean = path.replace(/^\//, "");
  // Static API JSON files have fixed names (no content hash), so the browser/CDN
  // can serve a stale copy after a redeploy. __BUILD_ID__ changes every build and
  // busts that cache without affecting the hashed JS/CSS bundles.
  return `${import.meta.env.BASE_URL}${clean}.json?v=${__BUILD_ID__}`;
}

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function pct(x: number | undefined | null, digits = 2) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(digits)}%`;
}

function num(x: number | undefined | null, digits = 3) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

function worstSingleDays(rets: number[], n: number) {
  return rets
    .map((r, i) => ({ index: i, depth: r }))
    .filter((d) => d.depth < 0)
    .sort((a, b) => a.depth - b.depth)
    .slice(0, n);
}

function periodSummary(rows: MonthlyRow[]) {
  if (!rows.length) return null;
  const months = rows.map((r) => r.month).sort();
  const qs = Array.from(
    new Set(rows.map((r) => r.signal_quarter).filter(Boolean) as string[])
  ).sort();
  return {
    nMonths: rows.length,
    mStart: months[0],
    mEnd: months[months.length - 1],
    nQ: qs.length,
    qStart: qs[0],
    qEnd: qs[qs.length - 1],
  };
}

function Kpi({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "pos" | "neg" | "neutral";
}) {
  return (
    <div className={`kpi ${tone ?? "neutral"}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

function ConfusionMatrix({ title, cm, topN }: { title: string; cm?: Cm; topN: number }) {
  if (!cm || cm.tp === undefined) return null;
  return (
    <div className="cm-block">
      <div className="cm-title">{title}</div>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
        <table className="cm-table">
          <thead>
            <tr>
              <th />
              <th>預測 非Top{topN}</th>
              <th>預測 Top{topN}</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>實際 非Top{topN}</th>
              <td className="cm-cell cm-tn">{cm.tn}</td>
              <td className="cm-cell cm-fp">{cm.fp}</td>
            </tr>
            <tr>
              <th>實際 Top{topN}</th>
              <td className="cm-cell cm-fn">{cm.fn}</td>
              <td className="cm-cell cm-tp">{cm.tp}</td>
            </tr>
          </tbody>
        </table>
        <div className="cm-stats">
          <div className="cm-stat">
            <span className="k">Precision</span>
            <span className="v">{pct(cm.precision, 1)}</span>
          </div>
          <div className="cm-stat">
            <span className="k">Recall</span>
            <span className="v">{pct(cm.recall, 1)}</span>
          </div>
          <div className="cm-stat">
            <span className="k">F1</span>
            <span className="v">{pct(cm.f1, 1)}</span>
          </div>
          <div className="cm-stat">
            <span className="k">Accuracy</span>
            <span className="v">{pct(cm.accuracy, 1)}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

type Tab = "overview" | "variables" | "method" | "holdings" | "factor";

export default function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  const [signalQuarters, setSignalQuarters] = useState<string[]>([]);
  const [selectedSignalQ, setSelectedSignalQ] = useState("");
  const [chartMode, setChartMode] = useState<"oos" | "is" | "both">("both");
  const [monthlyView, setMonthlyView] = useState<"oos" | "is">("oos");
  const [monthlyIs, setMonthlyIs] = useState<MonthlyRow[]>([]);
  const [monthlyOos, setMonthlyOos] = useState<MonthlyRow[]>([]);
  const [dailyIs, setDailyIs] = useState<DailyRow[]>([]);
  const [dailyOos, setDailyOos] = useState<DailyRow[]>([]);
  const [ddBreak, setDdBreak] = useState<DrawdownBreakdowns>({});
  const [report, setReport] = useState<PerfReport | null>(null);
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [holdings, setHoldings] = useState<HoldingsQuarter | null>(null);
  const [factor, setFactor] = useState<FactorAnalysis | null>(null);
  const [variables, setVariables] = useState<ModelVariables | null>(null);
  const [news, setNews] = useState<Record<string, { event: string; detail: string; url?: string; source?: string }>>({});
  const [selectedDD, setSelectedDD] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        const [r, mo, dy, ms, sq, fa] = await Promise.all([
          fetchJson<PerfReport>("/api/backtest/report"),
          fetchJson<{ monthly_in_sample: MonthlyRow[]; monthly_out_of_sample: MonthlyRow[] }>(
            "/api/backtest/monthly"
          ),
          fetchJson<{
            daily_in_sample: DailyRow[];
            daily_out_of_sample: DailyRow[];
            drawdowns?: DrawdownBreakdowns;
          }>("/api/backtest/daily"),
          fetchJson<ModelSummary>("/api/model/summary"),
          fetchJson<{ signal_quarters: string[] }>("/api/holdings/signal-quarters"),
          fetchJson<FactorAnalysis>("/api/factor/analysis"),
        ]);
        fetchJson<Record<string, { event: string; detail: string; url?: string; source?: string }>>(
          "/api/backtest/drawdown-news"
        )
          .then(setNews)
          .catch(() => setNews({}));
        fetchJson<ModelVariables>("/api/model/variables")
          .then(setVariables)
          .catch(() => setVariables(null));
        const qs = sq.signal_quarters ?? [];
        setSignalQuarters(qs);
        setSelectedSignalQ((prev) => prev || qs[qs.length - 1] || "");
        setReport(r);
        setMonthlyIs(mo.monthly_in_sample ?? []);
        setMonthlyOos(mo.monthly_out_of_sample ?? []);
        setDailyIs(dy.daily_in_sample ?? []);
        setDailyOos(dy.daily_out_of_sample ?? []);
        setDdBreak(dy.drawdowns ?? {});
        setModel(ms);
        setFactor(fa);
        setError(null);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedSignalQ) return;
    (async () => {
      try {
        const h = await fetchJson<HoldingsQuarter>(`/api/holdings/signal/${selectedSignalQ}`);
        setHoldings(h);
      } catch {
        setHoldings(null);
      }
    })();
  }, [selectedSignalQ]);

  const actualSet = useMemo(
    () => new Set(holdings?.actual_top30?.map((a) => a.symbol) ?? []),
    [holdings]
  );

  const selected = holdings?.selected ?? [];
  const hitCount = holdings?.hit_count ?? selected.filter((s) => actualSet.has(s.symbol)).length;
  const hitRate = holdings?.hit_rate ?? (selected.length ? hitCount / selected.length : 0);
  const realizedQ = holdings?.realized_quarter ?? "—";

  const chartSeries = useMemo(() => {
    const toPoints = (rows: DailyRow[]) =>
      rows.map((r) => ({ month: r.date, strategy: r.strategy_nav, benchmark: r.benchmark_nav }));
    if (chartMode === "is") return toPoints(dailyIs);
    if (chartMode === "oos") return toPoints(dailyOos);
    // 合併：OOS NAV 接續在 CV OOF 末端，避免從 1.0 重啟造成假性斷崖
    const isPts = toPoints(dailyIs);
    const oosPts = toPoints(dailyOos);
    if (!isPts.length) return oosPts;
    const sOff = isPts[isPts.length - 1].strategy;
    const bOff = isPts[isPts.length - 1].benchmark;
    const oosChained = oosPts.map((p) => ({
      month: p.month,
      strategy: p.strategy * sOff,
      benchmark: p.benchmark * bOff,
    }));
    return [...isPts, ...oosChained];
  }, [chartMode, dailyIs, dailyOos]);

  const chartDailyReturns = useMemo(() => {
    const ret = (rows: DailyRow[]) => rows.map((r) => r.daily_return);
    if (chartMode === "is") return ret(dailyIs);
    if (chartMode === "oos") return ret(dailyOos);
    return [...ret(dailyIs), ...ret(dailyOos)];
  }, [chartMode, dailyIs, dailyOos]);
  const chartDrawdowns = useMemo(
    () => worstSingleDays(chartDailyReturns, 5),
    [chartDailyReturns]
  );
  const ddByDate = useMemo(() => {
    const list =
      chartMode === "is"
        ? ddBreak.cv_oof
        : chartMode === "oos"
          ? ddBreak.final_oos
          : ddBreak.merged;
    const m = new Map<string, DDBreakdown>();
    (list ?? []).forEach((b) => m.set(b.date, b));
    return m;
  }, [chartMode, ddBreak]);
  const dividerIndex = chartMode === "both" && dailyIs.length ? dailyIs.length : null;

  const scatterPoints = useMemo(
    () =>
      selected
        .filter((s) => s.realized_excess !== undefined && s.realized_excess !== null)
        .map((s) => ({
          x: s.score,
          y: s.realized_excess as number,
          hit: !!s.is_top30_actual,
          symbol: s.symbol,
        })),
    [selected]
  );

  if (loading) {
    return (
      <div className="shell">
        <div className="loader">載入投組資料中…</div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="shell">
        <div className="error-box">載入失敗：{error}</div>
      </div>
    );
  }

  const is = report?.in_sample ?? {};
  const oos = report?.out_of_sample ?? {};
  const cvEv = model?.cv_eval;
  const oosEv = model?.oos_eval;
  const pool = model?.hp_pool;
  const bench = report?.benchmark_symbol ?? model?.portfolio?.benchmark_symbol ?? "SPY";
  const topN = report?.top_n_holdings ?? model?.portfolio?.top_n_holdings ?? 30;
  const monthlyRows = monthlyView === "oos" ? monthlyOos : monthlyIs;
  const isPeriod = periodSummary(monthlyIs);
  const oosPeriod = periodSummary(monthlyOos);
  const universe = model?.universe;
  const evalQuarters = Array.from(
    new Set(
      [...monthlyIs, ...monthlyOos].map((r) => r.signal_quarter).filter(Boolean) as string[]
    )
  );
  const evalUnivCounts = evalQuarters
    .map((q) => universe?.per_quarter?.[q])
    .filter((n): n is number => typeof n === "number");
  const univMin = evalUnivCounts.length ? Math.min(...evalUnivCounts) : undefined;
  const univMax = evalUnivCounts.length ? Math.max(...evalUnivCounts) : undefined;

  const navItems: Array<{ id: Tab; label: string }> = [
    { id: "overview", label: "績效總覽" },
    { id: "variables", label: "模型變數" },
    { id: "method", label: "驗證方法" },
    { id: "holdings", label: "持股對照" },
    { id: "factor", label: "因子分析" },
  ];
  const tabTitle: Record<Tab, string> = {
    overview: "績效總覽",
    variables: "模型變數",
    method: "驗證方法",
    holdings: "持股對照",
    factor: "因子分析",
  };

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">Q</div>
          <div>
            <div className="brand-title">Quant Dashboard</div>
            <div className="brand-sub">
              Top{topN} · {bench} · 月度調倉
            </div>
          </div>
        </div>
        <nav className="nav">
          {navItems.map((n) => (
            <button
              key={n.id}
              type="button"
              className={`nav-item ${tab === n.id ? "active" : ""}`}
              onClick={() => setTab(n.id)}
            >
              {n.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <div className="label">模型</div>
          <div className="small">{model?.best_estimator ?? "—"}</div>
          <div className="small">
            pool #{pool?.selected_pool_rank ?? "—"} · 權重 {pool?.weight_spec_label ?? "—"}
          </div>
          <div className="label" style={{ marginTop: 12 }}>
            調倉
          </div>
          <div className="small">季頻選股訊號 · 每月重設回目標權重</div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
        <div>
            <h1>{tabTitle[tab]}</h1>
            <p>
              Hyperparam pool top-{pool?.top_k ?? 5} · Top{topN} by score · 權重{" "}
              {pool?.weight_spec_label ?? "—"} · 基準 <strong>{bench}</strong>
            </p>
          </div>
          {tab === "holdings" && (
            <div className="topbar-actions dual">
              <div>
                <label className="label">信號季（命中對照）</label>
                <select value={selectedSignalQ} onChange={(e) => setSelectedSignalQ(e.target.value)}>
                  {signalQuarters.map((q) => (
                    <option key={q} value={q}>
                      {q} → 下季
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </header>

        {tab === "overview" && (
          <>
            <section className="panel">
              <div className="panel-head">
                <h2>資料期間</h2>
                <p className="muted">月度調倉、季頻選股訊號；下列為各評估區段的實際回測期間</p>
              </div>
              <div className="factor-alpha" style={{ marginTop: 12 }}>
                <div className="alpha-card">
                  <div className="seg-name">CV OOF（樣本內選型）</div>
                  <div className="alpha-v" style={{ color: "var(--accent)", fontSize: 18 }}>
                    {isPeriod ? `${isPeriod.nMonths} 個月 · ${isPeriod.nQ} 季` : "—"}
                  </div>
                  <div className="alpha-meta">
                    {isPeriod
                      ? `月度 ${isPeriod.mStart} ~ ${isPeriod.mEnd}　|　信號季 ${isPeriod.qStart} ~ ${isPeriod.qEnd}`
                      : "—"}
                  </div>
                </div>
                <div className="alpha-card">
                  <div className="seg-name">Final OOS（樣本外）</div>
                  <div className="alpha-v" style={{ color: "var(--accent)", fontSize: 18 }}>
                    {oosPeriod ? `${oosPeriod.nMonths} 個月 · ${oosPeriod.nQ} 季` : "—"}
                  </div>
                  <div className="alpha-meta">
                    {oosPeriod
                      ? `月度 ${oosPeriod.mStart} ~ ${oosPeriod.mEnd}　|　信號季 ${oosPeriod.qStart} ~ ${oosPeriod.qEnd}`
                      : "—"}
                  </div>
                </div>
                <div className="alpha-card">
                  <div className="seg-name">股票池</div>
                  <div className="alpha-v" style={{ color: "var(--accent)", fontSize: 18 }}>
                    {universe?.n_sp500_constituents
                      ? `${universe.n_sp500_constituents} → ${universe?.n_companies ?? "—"} 家`
                      : universe?.n_companies
                        ? `${universe.n_companies} 家公司`
                        : "—"}
                  </div>
                  <div className="alpha-meta">
                    {universe?.n_sp500_constituents
                      ? `S&P 500 名單 ${universe.n_sp500_constituents} 檔 → 實際可投資 ${universe?.n_companies} 家`
                      : ""}
                    {univMin && univMax
                      ? `${universe?.n_sp500_constituents ? "・" : ""}各評估季候選 ${univMin}~${univMax} 檔 · 選 Top${topN}`
                      : ` · 模型選 Top${topN}`}
                  </div>
                </div>
              </div>
              {universe?.coverage_note && (
                <div className="note-box" style={{ marginTop: 12 }}>
                  <strong>
                    為什麼是 {universe?.n_companies} 家、而不是 S&P 500 的{" "}
                    {universe?.n_sp500_constituents ?? 500} 家？
                  </strong>
                  <br />
                  {universe.coverage_note}
                </div>
              )}
            </section>

            <div className="section-title">CV OOF（樣本內選型）</div>
            <section className="kpi-grid">
              <Kpi label="年化報酬" value={pct(is.annualized_strategy as number)} tone="pos"
                sub={`${bench} ${pct(is.annualized_benchmark as number)}`} />
              <Kpi label="年化超額" value={pct(is.annualized_excess as number)} tone="pos" />
              <Kpi label="Sharpe（策略）" value={num(is.ann_sharpe_strategy as number, 2)}
                sub={`超額 ${num(is.ann_sharpe_excess as number, 2)} · ${bench} ${num(is.ann_sharpe_benchmark as number, 2)}`} />
              <Kpi label="最大單日回撤" value={pct(is.max_drawdown as number)} tone="neg"
                sub={
                  is.max_drawdown_date
                    ? `${is.max_drawdown_date} · 峰谷 ${pct(is.max_drawdown_peak_to_trough as number)}`
                    : undefined
                } />
            </section>

            <div className="section-title">Final OOS（樣本外）</div>
            <section className="kpi-grid">
              <Kpi label="年化報酬" value={pct(oos.annualized_strategy as number)}
                tone={(oos.annualized_strategy as number) >= 0 ? "pos" : "neg"}
                sub={`${bench} ${pct(oos.annualized_benchmark as number)}`} />
              <Kpi label="年化超額" value={pct(oos.annualized_excess as number)}
                tone={(oos.annualized_excess as number) >= 0 ? "pos" : "neg"} />
              <Kpi label="Sharpe（策略）" value={num(oos.ann_sharpe_strategy as number, 2)}
                sub={`超額 ${num(oos.ann_sharpe_excess as number, 2)} · ${bench} ${num(oos.ann_sharpe_benchmark as number, 2)}`} />
              <Kpi label="最大單日回撤" value={pct(oos.max_drawdown as number)} tone="neg"
                sub={
                  oos.max_drawdown_date
                    ? `${oos.max_drawdown_date} · 峰谷 ${pct(oos.max_drawdown_peak_to_trough as number)}`
                    : undefined
                } />
            </section>

            <section className="panel">
              <div className="panel-toolbar">
                <div className="seg">
                  {(["both", "oos", "is"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={chartMode === m ? "seg-btn active" : "seg-btn"}
                      onClick={() => setChartMode(m)}
                    >
                      {m === "both" ? "合併" : m === "oos" ? "Final OOS" : "CV OOF"}
                    </button>
                  ))}
                </div>
              </div>
              <PerformanceChart
                title={
                  chartMode === "oos"
                    ? "累積 NAV — Final OOS"
                    : chartMode === "is"
                      ? "累積 NAV — CV OOF"
                      : "累積 NAV — CV OOF + Final OOS"
                }
                series={chartSeries}
                benchmarkLabel={bench}
                dividerIndex={dividerIndex}
                drawdowns={chartDrawdowns}
                onMarkerClick={(d) => setSelectedDD(d)}
              />
              <p className="muted" style={{ marginTop: 8 }}>
                y 軸為 NAV（累積淨值）：期初投入 1 元的成長倍數，起點＝1.0；此為
                <strong>每日</strong>複利曲線
                <code style={{ margin: "0 4px" }}>NAV × (1 + 當日報酬)</code>。
                例：1.41＝累積 +41%、0.95＝−5%。黃虛線為 {bench} 同口徑 NAV。滑鼠移到線上可看當日數值。
              </p>
              {chartDrawdowns.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <div className="section-title" style={{ margin: "0 0 8px" }}>
                    前五大單日回撤・對應事件與當日個股表現（點擊紅圈或下列項目展開）
                  </div>
                  <p className="muted" style={{ margin: "0 0 8px" }}>
                    投組為 Top30 但集中在相關性高的高 beta 成長/AI/電力股，故特定總經或題材利空日會整體同跌；
                    下表為當日各持股的權重、當日報酬與對投組的貢獻（權重×報酬），由跌幅貢獻最大排到最小。
                  </p>
                  <div className="dd-list">
                    {chartDrawdowns.map((d) => {
                      const date = chartSeries[d.index]?.month;
                      const n = date ? news[date] : undefined;
                      const bd = date ? ddByDate.get(date) : undefined;
                      const active = selectedDD === date;
                      return (
                        <button
                          type="button"
                          key={date}
                          className={`dd-item ${active ? "active" : ""}`}
                          onClick={() => setSelectedDD(active ? null : date)}
                        >
                          <div className="dd-row">
                            <span className="dd-date">{date}</span>
                            <span className="num-neg dd-pct">{pct(d.depth)}</span>
                            <span className="dd-event">{n?.event ?? "（無對應新聞註記）"}</span>
                          </div>
                          {active && (
                            <div className="dd-detail">
                              {n && (
                                <p style={{ margin: "0 0 8px" }}>
                                  {n.detail}
                                  {n.url && (
                                    <>
                                      {" "}
                                      <a href={n.url} target="_blank" rel="noreferrer">
                                        新聞來源{n.source ? `（${n.source}）` : ""} ↗
                                      </a>
                                    </>
                                  )}
                                </p>
                              )}
                              {bd ? (
                                <div className="dd-stocks-wrap">
                                  <div className="dd-stocks-head">
                                    當日投組 {pct(bd.portfolio_return)}・持有 {bd.n_held} 檔
                                  </div>
                                  <table className="dd-stocks">
                                    <thead>
                                      <tr>
                                        <th>股票</th>
                                        <th>權重</th>
                                        <th>當日報酬</th>
                                        <th>貢獻</th>
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {bd.holdings.map((h) => (
                                        <tr key={h.symbol}>
                                          <td className="sym">{h.symbol}</td>
                                          <td>{pct(h.weight)}</td>
                                          <td className={h.daily_ret < 0 ? "num-neg" : ""}>
                                            {pct(h.daily_ret)}
                                          </td>
                                          <td className={h.contrib < 0 ? "num-neg" : ""}>
                                            {pct(h.contrib)}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              ) : (
                                <p className="muted" style={{ margin: 0 }}>（當日個股明細無資料）</p>
                              )}
                            </div>
                          )}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </section>

            <section className="panel">
              <div className="panel-head">
                <h2>Top{topN} 命中 Confusion Matrix（每季分數 Top{topN} vs 實際超額 Top{topN}）</h2>
              </div>
              <div className="cm-grid" style={{ marginTop: 12 }}>
                <ConfusionMatrix title="CV OOF" cm={cvEv?.confusion_matrix} topN={topN} />
                <ConfusionMatrix title="Final OOS" cm={oosEv?.confusion_matrix} topN={topN} />
              </div>
            </section>

            <section className="panel">
              <div className="panel-toolbar">
                <div className="seg">
                  {(["oos", "is"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={monthlyView === m ? "seg-btn active" : "seg-btn"}
                      onClick={() => setMonthlyView(m)}
                    >
                      {m === "oos" ? "Final OOS" : "CV OOF"}
                    </button>
                  ))}
                </div>
              </div>
              <div className="panel-head">
                <h2>月度報酬（{monthlyView === "oos" ? "Final OOS" : "CV OOF"}）</h2>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>月份</th>
                      <th>策略</th>
                      <th>{bench}</th>
                      <th>超額</th>
                      <th>組合波動</th>
                    </tr>
                  </thead>
                  <tbody>
                    {monthlyRows.map((r) => (
                      <tr key={r.month}>
                        <td>{r.month}</td>
                        <td className={r.portfolio_return >= 0 ? "num-pos" : "num-neg"}>
                          {pct(r.portfolio_return)}
                        </td>
                        <td>{pct(r.benchmark_return)}</td>
                        <td className={r.excess_return >= 0 ? "num-pos" : "num-neg"}>
                          {pct(r.excess_return)}
                        </td>
                        <td>{pct(r.portfolio_vol)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <p className="muted" style={{ marginTop: -4 }}>
              模型選型方法（超參池 × 權重的 CV OOF 選擇、walk-forward 流程圖）請見「驗證方法」分頁。
            </p>
          </>
        )}

        {tab === "variables" && <VariablesView data={variables} />}
        {tab === "method" && <MethodologyView model={model} />}

        {tab === "holdings" && (
          <>
            <section className="kpi-grid">
              <Kpi label={`${selectedSignalQ}→${realizedQ} 命中`} value={`${hitCount} / ${selected.length}`}
                sub={`季頻 Top${topN}（對齊 y_next）${pct(hitRate, 1)}`}
                tone={hitRate >= 0.3 ? "pos" : "neutral"} />
              <Kpi label="CV OOF Spearman" value={num(cvEv?.spearman_mean, 3)} />
              <Kpi label="Final OOS Spearman" value={num(oosEv?.spearman_mean, 3)} />
              <Kpi label="OOS Top30 Precision" value={pct(oosEv?.top30_precision, 1)} />
            </section>

            <section className="panel">
              <div className="panel-head">
                <h2>預測分數 vs 真實季超額（信號 {selectedSignalQ} → 實現 {realizedQ}）</h2>
                <p className="muted">
                  每點為一檔選股：x = 模型分數 P(Top{topN})，y = 下一季實際超額報酬；綠=實際進 Top{topN}
                </p>
              </div>
              <ScatterChart points={scatterPoints} xLabel="預測分數" yLabel="真實季超額" />
            </section>

            <section className="panel">
              <div className="panel-head split">
                <div>
                  <h2>
                    持股對照 · 信號 <code>{selectedSignalQ}</code> → 實現 <code>{realizedQ}</code>
                  </h2>
                  <p className="muted">
                    左：信號季末選 <strong>Top{topN}</strong>（預測權重 / 分數 / 真實季報酬）｜右：
                    <strong>下一季</strong>實際超額 Top{topN}（對齊 <code>y_next</code>）
                  </p>
                </div>
                <div className="hit-badge">
                  命中 <strong>{hitCount}</strong> / {selected.length}
                </div>
              </div>

              <div className="holdings-grid">
                <div className="table-wrap">
                  <h3>模型選股 Top{topN}</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>代碼</th>
                        <th>權重</th>
                        <th>P(Top{topN})</th>
                        <th>真實季報酬</th>
                        <th>真實超額</th>
                        <th>命中</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.map((h, i) => {
                        const hit = h.is_top30_actual ?? actualSet.has(h.symbol);
                        return (
                          <tr key={h.symbol} className={hit ? "row-hit" : ""}>
                            <td>{i + 1}</td>
                            <td className="sym">{h.symbol}</td>
                            <td>{(h.weight * 100).toFixed(2)}%</td>
                            <td>{(h.score * 100).toFixed(2)}%</td>
                            <td className={(h.realized_return ?? 0) >= 0 ? "num-pos" : "num-neg"}>
                              {pct(h.realized_return)}
                            </td>
                            <td className={(h.realized_excess ?? 0) >= 0 ? "num-pos" : "num-neg"}>
                              {pct(h.realized_excess)}
                            </td>
                            <td>
                              {hit ? (
                                <span className="pill hit">Top{topN}</span>
                              ) : (
                                <span className="pill miss">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="table-wrap">
                  <h3>實際超額 Top{topN}（{realizedQ} 季）</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>代碼</th>
                        <th>超額</th>
                        <th>季報酬</th>
                        <th>入選模型?</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(holdings?.actual_top30 ?? []).map((a, i) => {
                        const picked = selected.some((s) => s.symbol === a.symbol);
                        return (
                          <tr key={a.symbol} className={picked ? "row-hit" : ""}>
                            <td>{i + 1}</td>
                            <td className="sym">{a.symbol}</td>
                            <td className={a.excess_return >= 0 ? "num-pos" : "num-neg"}>
                              {pct(a.excess_return)}
                            </td>
                            <td>{pct(a.return)}</td>
                            <td>
                              {picked ? (
                                <span className="pill hit">已選</span>
                              ) : (
                                <span className="pill miss">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          </>
        )}

        {tab === "factor" && <FactorView factor={factor} />}
      </main>
    </div>
  );
}

function FactorView({ factor }: { factor: FactorAnalysis | null }) {
  if (!factor || !factor.segments) {
    return (
      <section className="panel">
        <p className="muted">尚無因子分析資料，請先執行 scripts/factor_analysis.py。</p>
      </section>
    );
  }
  const order: Array<{ key: string; label: string }> = [
    { key: "cv_oof", label: "CV OOF（樣本內）" },
    { key: "final_oos", label: "Final OOS（樣本外）" },
    { key: "all", label: "全期間" },
  ];
  const segs = factor.segments;
  const maxBeta = Math.max(
    1,
    ...order.flatMap(({ key }) => (segs[key]?.factors ?? []).map((f) => Math.abs(f.beta)))
  );

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>{factor.model}</h2>
          <p className="muted">
            {factor.source} · 因子區間 {factor.factor_data_window?.start} ~{" "}
            {factor.factor_data_window?.end}
          </p>
        </div>
        <div className="factor-alpha" style={{ marginTop: 14 }}>
          {order.map(({ key, label }) => {
            const s = segs[key];
            if (!s) return null;
            const sig = Math.abs(s.alpha_t_stat ?? 0) >= 1.96;
            return (
              <div className="alpha-card" key={key}>
                <div className="seg-name">
                  {label}（{s.n_months} 個月
                  {s.months && s.months.length
                    ? `：${s.months[0]} ~ ${s.months[s.months.length - 1]}`
                    : ""}
                  ）
                </div>
                <div className="alpha-v">
                  {pct(s.alpha_annualized_compound)}
                  <span className={`sig-badge ${sig ? "sig-yes" : "sig-no"}`}>
                    {sig ? "顯著 95%" : "不顯著"}
                  </span>
                </div>
                <div className="alpha-meta">
                  年化(複利) · 月 alpha {pct(s.alpha_monthly)} · t={num(s.alpha_t_stat, 2)} · p=
                  {num(s.alpha_p_value, 3)}
                  <br />
                  R² {num(s.r_squared, 2)} · adj {num(s.adj_r_squared, 2)}
                </div>
              </div>
            );
          })}
        </div>
        <div className="note-box">
          回歸式：<code>{factor.regression_spec}</code>。Alpha 為扣除六因子曝險後的超額報酬（截距），
          年化以複利 <code>(1+月α)^12−1</code> 表示。
          <br />
          <strong>解讀注意</strong>：CV / OOS 單區段各只有 11–12 個月，卻要估 7 個參數（截距+6 因子），
          殘差自由度僅 4–5，標準誤大、估計不穩——故單區段 alpha 雖然數字大，<strong>t 值多半
          &lt; 1.96（不顯著）</strong>，僅合併全期間（23 個月）較接近顯著。請以「方向與因子曝險結構」
          為主，alpha 絕對值僅供參考。
        </div>
      </section>

      {order.map(({ key, label }) => {
        const s = segs[key];
        if (!s || !s.factors) return null;
        return (
          <section className="panel" key={key}>
            <div className="panel-head">
              <h2>因子曝險 — {label}</h2>
            </div>
            <div className="table-wrap" style={{ marginTop: 8 }}>
              <table>
                <thead>
                  <tr>
                    <th>因子</th>
                    <th>beta</th>
                    <th>曝險方向</th>
                    <th>t 值</th>
                    <th>p 值</th>
                  </tr>
                </thead>
                <tbody>
                  {s.factors.map((f) => {
                    const widthPct = (Math.abs(f.beta) / maxBeta) * 100;
                    const sig = f.p_value < 0.05;
                    return (
                      <tr key={f.factor}>
                        <td className="sym">{f.factor}</td>
                        <td className={f.beta >= 0 ? "num-pos" : "num-neg"}>{num(f.beta, 3)}</td>
                        <td className="bar-cell">
                          <span
                            className={`bar ${f.beta >= 0 ? "pos" : "neg"}`}
                            style={{ width: `${Math.max(widthPct, 3)}%` }}
                          />
                        </td>
                        <td>
                          {num(f.t_stat, 2)}
                          {sig && <span className="sig">*</span>}
                        </td>
                        <td>{num(f.p_value, 3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        );
      })}

      <section className="panel">
        <div className="panel-head">
          <h2>因子定義與來源</h2>
        </div>
        <div className="table-wrap" style={{ marginTop: 8 }}>
          <table>
            <thead>
              <tr>
                <th>因子</th>
                <th>定義</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(factor.factor_definitions ?? {}).map(([k, v]) => (
                <tr key={k}>
                  <td className="sym">{k}</td>
                  <td>{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="note-box" style={{ marginTop: 12 }}>
          五因子取自 Ken French <code>F-F_Research_Data_5_Factors_2x3</code>（市場、規模 SMB、價值 HML、
          獲利能力 RMW、投資 CMA、無風險 RF）；動量 <code>Mom</code> 取自 <code>F-F_Momentum_Factor</code>。
          皆為官方月頻百分比資料，下載後轉為小數並與策略月報酬對齊回歸。
        </div>
      </section>
    </>
  );
}
