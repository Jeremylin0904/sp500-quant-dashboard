import { useT } from "../i18n";

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

export function MethodologyView({ model }: { model: Model | null }) {
  const t = useT();
  const CATS = {
    warmup: { label: t("暖機剔除", "Warmup excluded"), cls: "qc-warmup" },
    trainonly: { label: t("僅作訓練", "Train only"), cls: "qc-train" },
    cv: { label: t("CV OOF 驗證（選型）", "CV OOF validation (selection)"), cls: "qc-cv" },
    oos: { label: t("Final OOS 測試（凍結後）", "Final OOS test (after freeze)"), cls: "qc-oos" },
  } as const;
  if (!model) {
    return (
      <section className="panel">
        <p className="muted">{t("尚無模型資料。", "No model data yet.")}</p>
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
  const weightLabel = pool?.weight_spec_label ?? t("選定配重方案", "selected weight scheme");
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
          title={t(`訓練 ${f.train_start} ~ ${f.train_end}（${f.n_train_quarters} 季）`, `Train ${f.train_start} ~ ${f.train_end} (${f.n_train_quarters} Q)`)}
        >
          {t("訓練", "Train")} ×{f.n_train_quarters}
        </span>
        <span
          className={`wf-pred ${isCv ? "cv" : "oos"}`}
          style={{ gridColumn: `${p + 1} / ${p + 2}` }}
          title={`${isCv ? t("驗證", "Validate") : t("測試", "Test")} ${f.pred_quarter} · AUC ${num(f.pred_roc_auc, 3)}`}
        >
          {isCv ? t("驗證", "Val") : t("測試", "Test")}
        </span>
      </div>
    );
  };

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>{t("投組怎麼來的（定義）", "How the portfolio is built (definition)")}</h2>
          <p className="muted">{t("從模型機率 → 選 Top30 → 配重，組成每季持倉", "Model probability → pick Top30 → weight → quarterly holdings")}</p>
        </div>
        <div className="note-box mv-portfolio">
          <p>
            {t("模型", "Model")} <code>f</code> {t("對每檔股票", "outputs, for each stock")} <code>i</code> {t("在第", "at the end of quarter")} <code>t</code>{" "}
            {t("季末、以該時點可得的特徵", ", using the features available at that point")}{" "}
            <code>x<sub>i,t</sub></code>{t("（point-in-time，不前視）輸出一個機率：", " (point-in-time, no look-ahead), a probability:")}
          </p>
          <p className="mv-formula-line">
            <code>
              f(x<sub>i,t</sub>) = p̂<sub>i</sub> = P( y<sub>i,t+1</sub> = 1 | x<sub>i,t</sub> )
            </code>
          </p>
          <p>
            {t("其中", "where")} <code>
              y<sub>i,t+1</sub> = 1
            </code>{" "}
            {t(
              "代表「下一季是否為超額報酬 Top30（相對 SPY）」。把當季所有股票依",
              "means \u201cwhether the stock is in next quarter's excess-return Top30 (vs SPY)\u201d. Sort all stocks that quarter by"
            )}{" "}
            <code>p̂</code> {t("由高到低排序，取", "from high to low, take the")} <strong>{t("前 30 名", "top 30")}</strong> {t("作為持股，再以選定的配重方案", "as holdings, then apply the selected weight scheme")}{" "}
            <strong>{weightLabel}</strong> {t("給定權重", "to set weights")} <code>
              w<sub>i</sub>
            </code> {t("並正規化，組成投組：", "and normalize, forming the portfolio:")}
          </p>
          <p className="mv-formula-line">
            <code>
              Top30<sub>t</sub> = argtop<sub>30</sub>( p̂<sub>·,t</sub> ) ， w<sub>i</sub> ≥ 0 ， Σ
              <sub>i∈Top30</sub> w<sub>i</sub> = 1
            </code>
          </p>
          <p>
            {t("每季末重算", "At each quarter-end we recompute")} <code>p̂</code>{t("、重選 Top30、重設權重（", ", re-pick Top30 and reset weights (")}<strong>{t("季度再平衡", "quarterly rebalance")}</strong>{t("），持有一季後再依新一季的預測調整。下方說明的就是我們如何挑出這個", "), hold for a quarter, then adjust by the next quarter's prediction. Below is how we pick this")} <code>f</code>{t("（模型 × 配重）。", " (model × weight scheme).")}
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>{t("目標函數（loss）與選型指標", "Loss function & selection metric")}</h2>
          <p className="muted">{t("「訓練單一模型」與「挑哪個模型×配重」用的是不同層級的標準", "Training a single model and choosing which model×weights use criteria at different levels")}</p>
        </div>
        <div className="note-box mv-portfolio">
          <p>
            <strong>{t("① 訓練 loss：加權二元交叉熵（weighted binary cross-entropy / log loss）。", "\u2460 Training loss: weighted binary cross-entropy (log loss).")}</strong>{" "}
            {t("每個樹模型（FLAML 在", "Each tree model (FLAML searches among")} <code>xgboost / xgb_limitdepth / lgbm</code> {t("間搜尋，", ", with")} <code>metric="log_loss"</code>{t("）最小化：", ") minimizes:")}
          </p>
          <p className="mv-formula-line">
            <code>
              L = − (1/N) Σ<sub>i</sub> w<sub>i</sub> [ y<sub>i</sub> ln p̂<sub>i</sub> + (1−y
              <sub>i</sub>) ln(1−p̂<sub>i</sub>) ]
            </code>
          </p>
          <p>
            {t("因 Top30 是少數類，用", "Since Top30 is the minority class, we")} <strong>{t("class weight 處理不平衡", "handle imbalance with class weights")}</strong>{t("：正類權重", ": positive-class weight")}{" "}
            <code>
              w<sub>+</sub> = n<sub>neg</sub> / n<sub>pos</sub>
            </code>
            {t("、負類", ", negative-class")} <code>
              w<sub>−</sub> = 1
            </code>{t("（每個訓練折各自計算）。AutoML 內部也以 log loss 在候選 config 間挑超參。", " (computed per training fold). AutoML also uses log loss to pick hyperparameters among candidate configs.")}
          </p>
          <p>
            <strong>{t("② 最終選型指標：投組 OOF 年化超額 Sharpe（", "\u2461 Final selection metric: portfolio OOF annualized excess Sharpe (")}<code>ann_sharpe_excess</code>{t("）。", ").")}</strong>{" "}
            {t("log loss 只決定「單一分類器訓得好不好」；但要在", "log loss only decides how well a single classifier trains; but to choose the final combination among")}{" "}
            <strong>{t(`top-${topK} 模型 × 8 種配重`, `top-${topK} models × 8 weight schemes`)}</strong> {t("中挑出最終組合，是用 CV OOF 投組的", "we compare the CV OOF portfolio's")}{" "}
            <strong>{t("年化超額 Sharpe", "annualized excess Sharpe")}</strong> {t("來比，而", ", and")}<strong>{t("不是", "not")}</strong> {t("log loss——因為我們的目標是投組績效，不是純分類準確度。", "log loss — because our objective is portfolio performance, not pure classification accuracy.")}
          </p>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>{t("驗證流程總覽", "Validation workflow overview")}</h2>
          <p className="muted">
            {t(
              "季頻選股 · 防前視（point-in-time）· 選型只看 CV OOF，最終再做凍結的樣本外 walk-forward",
              "Quarterly stock selection · no look-ahead (point-in-time) · selection only on CV OOF, then a frozen out-of-sample walk-forward"
            )}
          </p>
        </div>
        <div className="flow-steps">
          <div className="flow-step">
            <div className="flow-no">1</div>
            <div>
              <h3>{t("資料與 TTM 暖機", "Data & TTM warmup")}</h3>
              <p>
                {t("共", "Total")} <strong>{t(`${quarters.length || 20} 季`, `${quarters.length || 20} quarters`)}</strong>
                （{quarters[0]} ~ {quarters[quarters.length - 1]}）{t("。剔除前", ". Drop the first")}{" "}
                <strong>{t(`${wf.feature_warmup_quarters ?? 4} 季`, `${wf.feature_warmup_quarters ?? 4} quarters`)}</strong>（
                {(wf.feature_warmup_excluded ?? []).join("、")}）{t("：因 YoY 成長率與 TTM 比率需", ": because YoY growth and TTM ratios need")}{" "}
                <strong>{t("≥4 季歷史", "≥4 quarters of history")}</strong>{t("，這幾季特徵幾乎全是 NaN，不進 pool search / 訓練。", "; features in these quarters are almost all NaN, so they are excluded from pool search / training.")}
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">2</div>
            <div>
              <h3>{t(`Pool search（AutoML ${budget}s）`, `Pool search (AutoML ${budget}s)`)}</h3>
              <p>
                {t("在暖機後、樣本外前的", "Over the")} <strong>{t(`${(wf.selection_train_quarters ?? []).length} 季`, `${(wf.selection_train_quarters ?? []).length} quarters`)}</strong> {t("（", "(")}
                {wf.selection_train_quarters?.[0]} ~{" "}
                {wf.selection_train_quarters?.[(wf.selection_train_quarters?.length ?? 1) - 1]}{t("）跑 FLAML", ") after warmup and before OOS, run FLAML for")}{" "}
                {budget} {t("秒，取", "s and take the")} <strong>top-{topK}</strong> {t("個模型設定組成超參池（estimator + config）。", "model configs to form the hyperparameter pool (estimator + config).")}
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">3</div>
            <div>
              <h3>{t("Walk-forward CV OOF：選 model × 權重", "Walk-forward CV OOF: pick model × weights")}</h3>
              <p>
                {t("對 top-", "For top-")}{topK} {t("模型 ×", "models ×")} <strong>{t("8 種權重方案", "8 weight schemes")}</strong>{t("，用", ", use")}{" "}
                <strong>{t("expanding walk-forward", "expanding walk-forward")}</strong> {t("產生樣本外 OOF 預測（CV 折預測", "to produce out-of-fold predictions (CV folds predict")}{" "}
                {wf.cv_oof_quarters?.join("、")}{t("），組成投組後以", "), build a portfolio, then pick the best (model+weights) by")}{" "}
                <strong>{t("OOF 年化超額 Sharpe", "OOF annualized excess Sharpe")}</strong> {t("選出最佳 (模型+權重)，並", "and")} <strong>{t("凍結", "freeze it")}</strong>。
              </p>
            </div>
          </div>
          <div className="flow-step">
            <div className="flow-no">4</div>
            <div>
              <h3>{t("凍結 → Final OOS walk-forward", "Freeze → Final OOS walk-forward")}</h3>
              <p>
                {t("超參凍結後，對", "After freezing hyperparameters, for")} {wf.final_oos_quarters?.join("、")} {t("做", "do an")}{" "}
                <strong>{t("expanding 重新訓練 + 預測", "expanding retrain + predict")}</strong>{t("。這段完全沒參與任何選型，是真正的樣本外績效。", ". This segment is not involved in any selection — it is the genuine out-of-sample performance.")}
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>{t("季別時間軸", "Quarter timeline")}</h2>
          <p className="muted">{t("每格為一個季度；顏色代表該季在流程中的角色", "Each cell is a quarter; the color is its role in the workflow")}</p>
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
          <h2>{t(`Expanding walk-forward 折（${cvFolds.length} CV OOF + ${oosFolds.length} Final OOS）`, `Expanding walk-forward folds (${cvFolds.length} CV OOF + ${oosFolds.length} Final OOS)`)}</h2>
          <p className="muted">
            {t(
              "每折用「到該季前的所有可用季」做訓練（擴張視窗），下一季做評估；藍＝CV OOF（訓練→驗證，用於選型），綠＝Final OOS（訓練→測試，凍結後樣本外）",
              "Each fold trains on all quarters up to that point (expanding window) and evaluates the next quarter; blue = CV OOF (train → validate, used for selection), green = Final OOS (train → test, out-of-sample after freeze)"
            )}
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
          <h2>{t("Hyperparam Pool · model × 權重公式（CV OOF 選型）", "Hyperparam pool · model × weight scheme (CV OOF selection)")}</h2>
          <p className="muted">
            {t("選中", "Selected")} {pool?.selected_estimator ?? model.best_estimator}（rank #{pool?.selected_pool_rank}，
            {t("權重", "weights")} {pool?.weight_spec_label ?? "—"}）· {t("選型指標", "metric")}{" "}
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
                  <th>{t("權重", "Weights")}</th>
                  <th>{t("OOF 超額 Sharpe", "OOF excess Sharpe")}</th>
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
