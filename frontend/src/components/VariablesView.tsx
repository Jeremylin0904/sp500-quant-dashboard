import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Stat = {
  col: string;
  miss_pct?: number;
  inf_pct?: number;
  p01?: number;
  p50?: number;
  p99?: number;
  min?: number;
  max?: number;
};

type FeatureItem = {
  feature: string;
  formula: string;
  miss_pct?: number | null;
  inf_pct?: number | null;
};

type Group = { title: string; items: FeatureItem[]; kind?: "raw" | "engineered" };

export type ModelVariables = {
  markdown?: string;
  groups?: Group[];
  n_grouped_features?: number;
  not_in_model?: string[];
  feature_stats?: Stat[];
  feature_cols?: string[];
  n_features?: number;
  n_rows?: number;
  target_col?: string;
  best_estimator?: string;
};

function fmtNum(x: number | undefined | null) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return x.toExponential(2);
  return x.toFixed(a >= 100 ? 1 : 3);
}

function missClass(m?: number | null) {
  const v = m ?? 0;
  if (v >= 25) return "mv-miss hi";
  if (v >= 10) return "mv-miss mid";
  return "mv-miss";
}

const HIGHLIGHTS = [
  { k: "預測目標", v: "y_next＝下一季是否為超額 Top30（vs SPY）" },
  { k: "股票池", v: "S&P 500；基準 SPY（市值加權）" },
  { k: "對齊方式", v: "Point-in-time：只用 publish_date ≤ 季末的最新財報，不前視" },
  { k: "缺值處理", v: "不補值，樹模型走 missing branch；數值 clip 到 [-1e6, 1e6]" },
  { k: "資料來源", v: "SimFin 財報 + 股價，經 TTM/YoY/比率/產業排名加工" },
];

export function VariablesView({ data }: { data: ModelVariables | null }) {
  if (!data) {
    return (
      <section className="panel">
        <p className="muted">尚無模型變數資料。</p>
      </section>
    );
  }
  const groups = data.groups ?? [];
  const notInModel = data.not_in_model ?? [];
  const rawCount = groups
    .filter((g) => g.kind === "raw")
    .reduce((n, g) => n + g.items.length, 0);
  const engCount = (data.n_grouped_features ?? 0) - rawCount;
  const topMissing = (data.feature_stats ?? [])
    .filter((s) => (s.miss_pct ?? 0) > 0)
    .sort((a, b) => (b.miss_pct ?? 0) - (a.miss_pct ?? 0))
    .slice(0, 8);
  const maxMiss = topMissing.length ? topMissing[0].miss_pct ?? 1 : 1;
  const allStats = (data.feature_stats ?? [])
    .slice()
    .sort((a, b) => (b.miss_pct ?? 0) - (a.miss_pct ?? 0));

  return (
    <>
      <section className="kpi-grid">
        <div className="kpi neutral">
          <div className="kpi-label">真正進模型的特徵數</div>
          <div className="kpi-value">{data.n_features ?? "—"}</div>
          <div className="kpi-sub">
            {rawCount} 原始量 + {engCount} 工程衍生 · {groups.length} 類
          </div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">資料列數</div>
          <div className="kpi-value">{data.n_rows ?? "—"}</div>
          <div className="kpi-sub">股票 × 季度末（PIT）</div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">預測目標</div>
          <div className="kpi-value">{data.target_col ?? "y_next"}</div>
          <div className="kpi-sub">下一季超額 Top30 (0/1)</div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">模型</div>
          <div className="kpi-value">{data.best_estimator ?? "—"}</div>
          <div className="kpi-sub">FLAML AutoML</div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>重點摘要</h2>
        </div>
        <div className="mv-highlights">
          {HIGHLIGHTS.map((h) => (
            <div className="mv-hl" key={h.k}>
              <span className="mv-hl-k">{h.k}</span>
              <span className="mv-hl-v">{h.v}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>真正進模型的變數（model_meta.json → feature_cols）</h2>
          <p className="muted">
            這 {data.n_features ?? data.n_grouped_features} 個欄位才是模型實際 fit 的輸入：
            <strong>原始量</strong>＝直接餵入的財報/價格水準值，<strong>衍生</strong>＝由原始資料工程出的成長率／比率／分位／動量／波動。缺失率即時取自
            feature_stats.csv，紅＝偏高（多因需 ≥4 季歷史或財報本身缺漏）。
          </p>
        </div>
        <div className="mv-groups">
          {groups.map((g) => (
            <div className="mv-group" key={g.title}>
              <h3>
                {g.title}
                <span className={`mv-kind ${g.kind === "raw" ? "raw" : "eng"}`}>
                  {g.kind === "raw" ? "原始量" : "衍生"}
                </span>
                <span className="mv-count">{g.items.length}</span>
              </h3>
              <ul className="mv-flist">
                {g.items.map((it) => (
                  <li key={it.feature}>
                    <div className="mv-frow">
                      <code className="mv-fname">{it.feature}</code>
                      <span className={missClass(it.miss_pct)}>{(it.miss_pct ?? 0).toFixed(1)}%</span>
                    </div>
                    <div className="mv-formula">{it.formula}</div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {topMissing.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>缺失率最高的特徵</h2>
          </div>
          <div className="mv-missbars">
            {topMissing.map((s) => (
              <div className="mv-missbar" key={s.col}>
                <code className="mv-fname">{s.col}</code>
                <div className="mv-track">
                  <span
                    className="mv-fill"
                    style={{ width: `${Math.max(((s.miss_pct ?? 0) / maxMiss) * 100, 4)}%` }}
                  />
                </div>
                <span className="mv-pct">{(s.miss_pct ?? 0).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {notInModel.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>不進模型的欄位（僅存於 parquet）</h2>
            <p className="muted">
              這些只是識別碼／日期／標籤或會洩漏未來的目標欄位，<strong>不會</strong>當作特徵餵給模型
            </p>
          </div>
          <div className="mv-chips">
            {notInModel.map((c) => (
              <code className="mv-chip" key={c}>
                {c}
              </code>
            ))}
          </div>
        </section>
      )}

      <section className="panel">
        <details>
          <summary className="mv-summary">完整文件與欄位統計（公式、PIT、NA 處理、資料流、分位數）</summary>
          <div className="md-body" style={{ marginTop: 12 }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.markdown ?? ""}</ReactMarkdown>
          </div>
          {allStats.length > 0 && (
            <div className="table-wrap compact" style={{ marginTop: 12 }}>
              <table>
                <thead>
                  <tr>
                    <th>欄位</th>
                    <th>缺失%</th>
                    <th>inf%</th>
                    <th>p01</th>
                    <th>中位數</th>
                    <th>p99</th>
                    <th>min</th>
                    <th>max</th>
                  </tr>
                </thead>
                <tbody>
                  {allStats.map((s) => (
                    <tr key={s.col}>
                      <td className="sym">{s.col}</td>
                      <td className={(s.miss_pct ?? 0) > 20 ? "num-neg" : ""}>
                        {(s.miss_pct ?? 0).toFixed(2)}%
                      </td>
                      <td>{(s.inf_pct ?? 0).toFixed(2)}%</td>
                      <td>{fmtNum(s.p01)}</td>
                      <td>{fmtNum(s.p50)}</td>
                      <td>{fmtNum(s.p99)}</td>
                      <td>{fmtNum(s.min)}</td>
                      <td>{fmtNum(s.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </details>
      </section>
    </>
  );
}
