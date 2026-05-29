import { useState } from "react";

type Point = { month: string; strategy: number; benchmark: number };

type Props = {
  title: string;
  series: Point[];
  benchmarkLabel?: string;
  dividerIndex?: number | null;
  drawdowns?: { index: number; depth: number }[];
  onMarkerClick?: (date: string) => void;
};

export function PerformanceChart({
  title,
  series,
  benchmarkLabel = "SPY",
  dividerIndex = null,
  drawdowns = [],
  onMarkerClick,
}: Props) {
  const [hover, setHover] = useState<number | null>(null);

  if (!series.length) {
    return <div className="chart-empty">尚無曲線資料</div>;
  }

  const w = 920;
  const h = 320;
  const padL = 48;
  const padR = 16;
  const padT = 24;
  const padB = 40;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const vals = series.flatMap((p) => [p.strategy, p.benchmark]);
  // Always include the NAV=1.0 baseline, and snap to a tidy step so 1.00 lands on a gridline.
  const rawMin = Math.min(...vals, 1);
  const rawMax = Math.max(...vals, 1);
  const span = rawMax - rawMin || 0.1;
  const niceSteps = [0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
  const step = niceSteps.find((s) => s >= span / 5) ?? 1;
  const ymin = Math.floor(rawMin / step) * step;
  const ymax = Math.ceil(rawMax / step) * step;
  const yRange = ymax - ymin || 1;

  const x = (i: number) => padL + (i / Math.max(series.length - 1, 1)) * innerW;
  const y = (v: number) => padT + innerH - ((v - ymin) / yRange) * innerH;

  const line = (key: "strategy" | "benchmark") =>
    series
      .map((p, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(p[key]).toFixed(1)}`)
      .join(" ");

  const yTicks = Math.round(yRange / step);
  const gridLines = Array.from({ length: yTicks + 1 }, (_, i) => {
    const v = ymin + step * i;
    const yy = y(v);
    const isBase = Math.abs(v - 1) < 1e-9;
    return (
      <g key={i}>
        <line x1={padL} x2={w - padR} y1={yy} y2={yy} className="chart-grid" />
        <text
          x={padL - 8}
          y={yy + 4}
          textAnchor="end"
          className="chart-axis"
          style={isBase ? { fontWeight: 700, fill: "#475569" } : undefined}
        >
          {v.toFixed(2)}
        </text>
      </g>
    );
  });

  // NAV = 1.0 reference line (if within range)
  const oneLine =
    ymin <= 1 && ymax >= 1 ? (
      <line x1={padL} x2={w - padR} y1={y(1)} y2={y(1)} stroke="#cbd5e1" strokeWidth={1} strokeDasharray="3 3" />
    ) : null;

  const xLabels = series
    .filter((_, i) => i === 0 || i === series.length - 1 || i % Math.ceil(series.length / 6) === 0)
    .map((p) => {
      const i = series.findIndex((s) => s.month === p.month);
      return (
        <text key={p.month} x={x(i)} y={h - 12} textAnchor="middle" className="chart-axis">
          {p.month}
        </text>
      );
    });

  function handleMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const vbX = ((e.clientX - rect.left) / rect.width) * w;
    let idx = Math.round(((vbX - padL) / innerW) * (series.length - 1));
    idx = Math.max(0, Math.min(series.length - 1, idx));
    setHover(idx);
  }

  const hp = hover !== null ? series[hover] : null;
  const tipW = 150;
  const tipH = 70;
  const tipX = hover !== null ? (x(hover) > w - tipW - 10 ? x(hover) - tipW - 10 : x(hover) + 10) : 0;
  const tipY = padT + 4;

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <h3>{title}</h3>
        <div className="chart-legend">
          <span className="legend strategy">策略 NAV</span>
          <span className="legend benchmark">{benchmarkLabel}（市值加權）</span>
          {drawdowns.length > 0 && <span style={{ color: "#dc2626" }}>○ 前5大單日回撤</span>}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        className="chart-svg"
        role="img"
        aria-label={title}
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      >
        {gridLines}
        {oneLine}
        <path d={line("benchmark")} className="line-benchmark" fill="none" />
        <path d={line("strategy")} className="line-strategy" fill="none" />
        {series.length <= 60 &&
          series.map((p, i) => (
            <g key={p.month}>
              <circle cx={x(i)} cy={y(p.strategy)} r={3} className="dot-strategy" />
              <circle cx={x(i)} cy={y(p.benchmark)} r={2.5} className="dot-benchmark" />
            </g>
          ))}

        {dividerIndex !== null && dividerIndex > 0 && dividerIndex < series.length && (
          <g>
            <line
              x1={x(dividerIndex)}
              x2={x(dividerIndex)}
              y1={padT}
              y2={padT + innerH}
              stroke="#6366f1"
              strokeWidth={1.5}
              strokeDasharray="5 4"
            />
            <text x={x(dividerIndex) - 6} y={padT + 12} textAnchor="end" fontSize={11} fill="#6366f1">
              樣本內
            </text>
            <text x={x(dividerIndex) + 6} y={padT + 12} textAnchor="start" fontSize={11} fill="#6366f1">
              樣本外
            </text>
          </g>
        )}

        {drawdowns.map((d, k) =>
          d.index >= 0 && d.index < series.length ? (
            <g
              key={`dd-${k}`}
              onClick={() => onMarkerClick?.(series[d.index].month)}
              style={{ cursor: onMarkerClick ? "pointer" : "default" }}
            >
              <circle cx={x(d.index)} cy={y(series[d.index].strategy)} r={10} fill="transparent" />
              <circle
                cx={x(d.index)}
                cy={y(series[d.index].strategy)}
                r={5}
                fill="none"
                stroke="#dc2626"
                strokeWidth={2}
              />
              <text
                x={x(d.index)}
                y={y(series[d.index].strategy) + 18}
                textAnchor="middle"
                fontSize={10}
                fontWeight={700}
                fill="#dc2626"
              >
                {(d.depth * 100).toFixed(1)}%
              </text>
            </g>
          ) : null
        )}
        {xLabels}

        {hp && hover !== null && (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={padT}
              y2={padT + innerH}
              stroke="#94a3b8"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
            <circle cx={x(hover)} cy={y(hp.strategy)} r={5} fill="#15a34a" stroke="#fff" strokeWidth={1.5} />
            <circle cx={x(hover)} cy={y(hp.benchmark)} r={5} fill="#f59e0b" stroke="#fff" strokeWidth={1.5} />
            <g transform={`translate(${tipX}, ${tipY})`}>
              <rect
                width={tipW}
                height={tipH}
                rx={8}
                fill="#ffffff"
                stroke="#e2e8f0"
                strokeWidth={1}
                style={{ filter: "drop-shadow(0 2px 6px rgba(15,23,42,0.12))" }}
              />
              <text x={10} y={20} fontSize={12} fontWeight={700} fill="#0f172a">
                {hp.month}
              </text>
              <text x={10} y={40} fontSize={12} fill="#15a34a">
                策略 {hp.strategy.toFixed(3)}（{((hp.strategy - 1) * 100).toFixed(1)}%）
              </text>
              <text x={10} y={58} fontSize={12} fill="#b45309">
                {benchmarkLabel} {hp.benchmark.toFixed(3)}（{((hp.benchmark - 1) * 100).toFixed(1)}%）
              </text>
            </g>
          </g>
        )}
      </svg>
    </div>
  );
}
