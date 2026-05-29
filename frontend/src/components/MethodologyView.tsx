type Fold = {
  fold: number;
  phase: string;
  n_train_quarters?: number;
  train_start?: string;
  train_end?: string;
  pred_quarter?: string;
  pred_roc_auc?: number;
};

type WalkForward = {
  feature_warmup_quarters?: number;
  feature_warmup_excluded?: string[];
  selection_train_quarters?: string[];
  pool_search_time_budget_sec?: number;
  hp_pool_top_k?: number;
  n_cv_folds?: number;
  n_oos_folds?: number;
  folds?: Fold[];
  cv_oof_quarters?: string[];
  final_oos_quarters?: string[];
  min_train_quarters?: number;
};

type PoolEval = {
  pool_rank: number;
  estimator: string;
  weight_scheme?: string;
  weight_spec_label?: string;
  softmax_temperature?: number | null;
  mean_fold_log_loss?: number;
  oof_portfolio_ann_sharpe_excess?: number;
  pooled_oof_roc_auc?: number;
  selected?: boolean;
};

type Model = {
  best_estimator?: string;
  all_quarters?: string[];
  walk_forward?: WalkForward;
  hp_pool?: {
    top_k?: number;
    selected_pool_rank?: number;
    selected_estimator?: string;
    weight_spec_label?: string;
    portfolio_select_metric?: string;
    pool_search_time_budget_sec?: number;
    evaluations?: PoolEval[];
  };
  cv?: { oof_roc_auc?: number };
  final_oos?: { roc_auc?: number };
};

function num(x: number | undefined | null, digits = 3) {
  if (x === undefined || x === null || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

const CATS = {
  warmup: { label: "暖機剔除", cls: "qc-warmup" },
  trainonly: { label: "僅作訓練", cls: "qc-train" },
  cv: { label: "CV OOF 預測（選型）", cls: "qc-cv" },
  oos: { label: "Final OOS 預測（凍結後）", cls: "qc-oos" },
} as const;

export function MethodologyView({ model }: { model: Model | null }) {
  if (!model) {
    return (
      <section className="panel">
        <p className="muted">尚無模型資料。</p>
      </section>
    );
  }
  const wf = model.walk_forward ?? {};
  const quarters = model.all_quarters ?? [];
  const idx = new Map(quarters.map((q, i) => [q, i]));
  const warmup = new Set(wf.feature_warmup_excluded ?? []);
  const cv = new Set(wf.cv_oof_quarters ?? []);
  const oos = new Set(wf.final_oos_quarters ?? []);
  const catOf = (q: string) =>
    warmup.has(q) ? "warmup" : cv.has(q) ? "cv" : oos.has(q) ? "oos" : "trainonly";

  const pool = model.hp_pool;
  const topK = pool?.top_k ?? wf.hp_pool_top_k ?? 5;
  const budget = pool?.pool_search_time_budget_sec ?? wf.pool_search_time_budget_sec ?? 120;
  const folds = wf.folds ?? [];
  const cvFolds = folds.filter((f) => f.phase === "cv_oof");
  const oosFolds = folds.filter((f) => f.phase === "final_oos");
  const cols = quarters.length || 20;

  const FoldRow = ({ f }: { f: Fold }) => {
    const s = f.train_start ? idx.get(f.train_start) ?? 0 : 0;
    const e = f.train_end ? idx.get(f.train_end) ?? 0 : 0;
    const p = f.pred_quarter ? idx.get(f.pred_quarter) ?? 0 : 0;
    const isCv = f.phase === "cv_oof";
    return (
      <div className="wf-fold" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
        <span
          className={`wf-bar ${isCv ? "cv" : "oos"}`}
          style={{ gridColumn: `${s + 1} / ${e + 2}` }}
          title={`訓練 ${f.train_start} ~ ${f.train_end}（${f.n_train_quarters} 季）`}
        >
          train ×{f.n_train_quarters}
        </span>
        <span
          className={`wf-pred ${isCv ? "cv" : "oos"}`}
          style={{ gridColumn: `${p + 1} / ${p + 2}` }}
          title={`預測 ${f.pred_quarter} · AUC ${num(f.pred_roc_auc, 3)}`}
        >
          預測
        </span>
      </div>
    );
  };

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>驗證流程總覽</h2>
          <p className="muted">
            季頻選股 · 防前視（point-in-time）· 選型只看 CV OOF，最終再做凍結的樣本外 walk-forward
          </p>
        </div>
        <div className="flow-steps">
          <div className="flow-step">
            <div className="flow-no">1</div>
            <div>
              <h3>資料與 TTM 暖機</h3>
              <p>
                共 <strong>{quarters.length || 20} 季</strong>
                （{quarters[0]} ~ {quarters[quarters.length - 1]}）。剔除前{" "}
                <strong>{wf.feature_warmup_quarters ?? 4} 季</strong>（
                {(wf.feature_warmup_excluded ?? []).join("、")}）：因 YoY 成長率與 TTM 比率需{" "}
                <strong>≥4 季歷史</strong>，這幾季特徵幾乎全是 NaN，不進 pool search / 訓練。
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">2</div>
            <div>
              <h3>Pool search（AutoML {budget}s）</h3>
              <p>
                在暖機後、樣本外前的 <strong>{(wf.selection_train_quarters ?? []).length} 季</strong>（
                {wf.selection_train_quarters?.[0]} ~{" "}
                {wf.selection_train_quarters?.[(wf.selection_train_quarters?.length ?? 1) - 1]}）跑 FLAML{" "}
                {budget} 秒，取 <strong>top-{topK}</strong> 個模型設定組成超參池（estimator + config）。
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">3</div>
            <div>
              <h3>Walk-forward CV OOF：選 model × 權重</h3>
              <p>
                對 top-{topK} 模型 × <strong>8 種權重方案</strong>，用{" "}
                <strong>expanding walk-forward</strong> 產生樣本外 OOF 預測（CV 折預測{" "}
                {wf.cv_oof_quarters?.join("、")}），組成投組後以{" "}
                <strong>OOF 年化超額 Sharpe</strong> 選出最佳 (模型+權重)，並 <strong>凍結</strong>。
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">4</div>
            <div>
              <h3>凍結 → Final OOS walk-forward</h3>
              <p>
                超參凍結後，對 {wf.final_oos_quarters?.join("、")} 做{" "}
                <strong>expanding 重新訓練 + 預測</strong>。這段完全沒參與任何選型，是真正的樣本外績效。
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>季別時間軸</h2>
          <p className="muted">每格為一個季度；顏色代表該季在流程中的角色</p>
        </div>
        <div className="q-legend">
          {Object.values(CATS).map((c) => (
            <span key={c.label} className="q-leg-item">
              <span className={`q-chip ${c.cls}`} /> {c.label}
            </span>
          ))}
        </div>
        <div className="q-timeline" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
          {quarters.map((q) => (
            <div key={q} className={`q-cell ${CATS[catOf(q)].cls}`}>
              <span className="q-name">{q}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Expanding walk-forward 折（{cvFolds.length} CV OOF + {oosFolds.length} Final OOS）</h2>
          <p className="muted">
            每折用「到預測季前的所有可用季」做訓練（擴張視窗），只預測下一季；藍＝選型用 CV OOF，綠＝凍結後樣本外
          </p>
        </div>
        <div className="wf-grid">
          <div className="wf-axis" style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}>
            {quarters.map((q) => (
              <span key={q} className="wf-axis-q">
                {q.replace("20", "'")}
              </span>
            ))}
          </div>
          {[...cvFolds, ...oosFolds].map((f) => (
            <FoldRow key={f.fold} f={f} />
          ))}
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Hyperparam Pool · model × 權重公式（CV OOF 選型）</h2>
          <p className="muted">
            選中 {pool?.selected_estimator ?? model.best_estimator}（rank #{pool?.selected_pool_rank}，
            權重 {pool?.weight_spec_label ?? "—"}）· 選型指標{" "}
            {pool?.portfolio_select_metric ?? "ann_sharpe_excess"} · CV AUC {num(model.cv?.oof_roc_auc, 3)}{" "}
            · OOS AUC {num(model.final_oos?.roc_auc, 3)}
          </p>
        </div>
        {(pool?.evaluations?.length ?? 0) > 0 && (
          <div className="table-wrap compact" style={{ marginTop: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Estimator</th>
                  <th>權重</th>
                  <th>OOF 超額 Sharpe</th>
                  <th>log_loss</th>
                  <th>OOF AUC</th>
                </tr>
              </thead>
              <tbody>
                {pool!.evaluations!.map((e, i) => (
                  <tr
                    key={`${e.pool_rank}-${e.weight_scheme}-${e.softmax_temperature ?? i}`}
                    className={e.selected ? "row-active" : ""}
                  >
                    <td>{e.pool_rank}</td>
                    <td>{e.estimator}</td>
                    <td>{e.weight_spec_label ?? e.weight_scheme ?? "—"}</td>
                    <td>{num(e.oof_portfolio_ann_sharpe_excess, 2)}</td>
                    <td>{num(e.mean_fold_log_loss, 4)}</td>
                    <td>{num(e.pooled_oof_roc_auc, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
