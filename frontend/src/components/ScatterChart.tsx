type Pt = { x: number; y: number; hit: boolean; symbol: string };

type Props = {
  points: Pt[];
  xLabel: string;
  yLabel: string;
};

export function ScatterChart({ points, xLabel, yLabel }: Props) {
  if (!points.length) {
    return <div className="chart-empty">尚無資料</div>;
  }

  const w = 720;
  const h = 320;
  const padL = 56;
  const padR = 16;
  const padT = 18;
  const padB = 44;
  const innerW = w - padL - padR;
  const innerH = h - padT - padB;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const ymin = Math.min(...ys, 0);
  const ymax = Math.max(...ys, 0);
  const xpad = (xmax - xmin) * 0.08 || 0.01;
  const ypad = (ymax - ymin) * 0.08 || 0.01;
  const x0 = xmin - xpad;
  const x1 = xmax + xpad;
  const y0 = ymin - ypad;
  const y1 = ymax + ypad;

  const sx = (v: number) => padL + ((v - x0) / (x1 - x0 || 1)) * innerW;
  const sy = (v: number) => padT + innerH - ((v - y0) / (y1 - y0 || 1)) * innerH;

  const yTicks = 5;
  const grid = Array.from({ length: yTicks + 1 }, (_, i) => {
    const v = y0 + ((y1 - y0) * i) / yTicks;
    const yy = sy(v);
    return (
      <g key={i}>
        <line x1={padL} x2={w - padR} y1={yy} y2={yy} className="chart-grid" />
        <text x={padL - 8} y={yy + 4} textAnchor="end" className="chart-axis">
          {(v * 100).toFixed(0)}%
        </text>
      </g>
    );
  });

  const xTicks = 5;
  const xgrid = Array.from({ length: xTicks + 1 }, (_, i) => {
    const v = x0 + ((x1 - x0) * i) / xTicks;
    const xx = sx(v);
    return (
      <text key={i} x={xx} y={h - 14} textAnchor="middle" className="chart-axis">
        {(v * 100).toFixed(0)}%
      </text>
    );
  });

  const zeroY = sy(0);

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <h3>
          {xLabel} vs {yLabel}
        </h3>
        <div className="chart-legend">
          <span className="legend strategy">命中 Top30</span>
          <span className="legend benchmark">未命中</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg" role="img" aria-label="scatter">
        {grid}
        {xgrid}
        <line x1={padL} x2={w - padR} y1={zeroY} y2={zeroY} stroke="#cbd5e1" strokeWidth={1.2} />
        {points.map((p) => (
          <circle
            key={p.symbol}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={5}
            fill={p.hit ? "#15a34a" : "#f59e0b"}
            fillOpacity={0.75}
            stroke="#fff"
            strokeWidth={1}
          >
            <title>
              {p.symbol}: {xLabel} {(p.x * 100).toFixed(1)}% · {yLabel} {(p.y * 100).toFixed(1)}%
            </title>
          </circle>
        ))}
        <text
          x={padL + innerW / 2}
          y={h - 1}
          textAnchor="middle"
          className="chart-axis"
        >
          {xLabel}
        </text>
      </svg>
    </div>
  );
}
