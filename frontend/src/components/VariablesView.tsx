import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Stat = {
  col: string;
  n?: number;
  miss_pct?: number;
  inf_pct?: number;
  p01?: number;
  p50?: number;
  p99?: number;
  min?: number;
  max?: number;
};

export type ModelVariables = {
  markdown?: string;
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

export function VariablesView({ data }: { data: ModelVariables | null }) {
  if (!data) {
    return (
      <section className="panel">
        <p className="muted">尚無模型變數資料。</p>
      </section>
    );
  }
  const stats = (data.feature_stats ?? [])
    .slice()
    .sort((a, b) => (b.miss_pct ?? 0) - (a.miss_pct ?? 0));

  return (
    <>
      <section className="kpi-grid">
        <div className="kpi neutral">
          <div className="kpi-label">進模型特徵數</div>
          <div className="kpi-value">{data.n_features ?? "—"}</div>
          <div className="kpi-sub">tree model · 不補值，用 missing branch</div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">資料列數</div>
          <div className="kpi-value">{data.n_rows ?? "—"}</div>
          <div className="kpi-sub">股票 × 季度末（PIT）</div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">預測目標</div>
          <div className="kpi-value">{data.target_col ?? "y_next"}</div>
          <div className="kpi-sub">下一季是否為超額 Top30</div>
        </div>
        <div className="kpi neutral">
          <div className="kpi-label">最佳模型</div>
          <div className="kpi-value">{data.best_estimator ?? "—"}</div>
          <div className="kpi-sub">FLAML AutoML</div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>變數定義、公式與原始資料</h2>
          <p className="muted">以下內容由 quant/model/MODEL_VARIABLES.md 即時渲染（單一事實來源）</p>
        </div>
        <div className="md-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.markdown ?? ""}</ReactMarkdown>
        </div>
      </section>

      {stats.length > 0 && (
        <section className="panel">
          <div className="panel-head">
            <h2>欄位統計（缺失率 / inf / 分位數）</h2>
            <p className="muted">來源：feature_stats.csv（依缺失率排序）。缺值主因：需 4 季歷史的 YoY/TTM 與財報本身缺漏</p>
          </div>
          <div className="table-wrap compact" style={{ marginTop: 8 }}>
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
                {stats.map((s) => (
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
        </section>
      )}
    </>
  );
}
