"use client";

import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, Info } from "lucide-react";
import { ChartCard, ChartTooltip, axisProps, gridProps } from "@/components/charts/chart";
import { BAND_VAR, BandBadge, OutcomeBadge, Provenance, prov } from "@/components/ui/domain";
import { Card, CardBody, CardHeader, Stat } from "@/components/ui/primitives";
import type { DecideResponse, ScoreResponse } from "@/lib/api/hooks";
import { money, pct, shortId } from "@/lib/format";

const logit = (p: number) => Math.log(Math.min(Math.max(p, 1e-9), 1 - 1e-9) / (1 - Math.min(Math.max(p, 1e-9), 1 - 1e-9)));

/** Semicircle PD gauge with the band zones and the applied threshold. */
export function PdGauge({ pd, threshold, cutoffs }: { pd: number; threshold: number; cutoffs: number[] }) {
  const max = Math.max(0.5, Math.min(1, threshold * 2.5, pd * 1.25));
  const W = 220;
  const R = 90;
  const cx = W / 2;
  const cy = 104;
  const angle = (v: number) => Math.PI * (1 - Math.min(v, max) / max);
  const pt = (v: number, r = R) => [cx + r * Math.cos(angle(v)), cy - r * Math.sin(angle(v))];
  const arc = (a: number, b: number) => {
    const [x1, y1] = pt(a);
    const [x2, y2] = pt(b);
    return `M ${x1} ${y1} A ${R} ${R} 0 0 1 ${x2} ${y2}`;
  };
  const edges = [0, ...cutoffs.filter((c) => c < max), max];
  const labels = ["Low", "Moderate", "Elevated", "High"];
  const [tx1, ty1] = pt(threshold, R - 14);
  const [tx2, ty2] = pt(threshold, R + 10);
  const [nx, ny] = pt(pd, R - 22);
  return (
    <svg viewBox={`0 0 ${W} 120`} className="w-full max-w-[260px]" role="img" aria-label={`PD ${pct(pd)} against threshold ${pct(threshold)}`}>
      {edges.slice(0, -1).map((e, i) => (
        <path key={i} d={arc(e, edges[i + 1])} stroke={BAND_VAR[labels[i]]} strokeOpacity={0.35} strokeWidth={12} fill="none" />
      ))}
      <path d={arc(0, Math.min(pd, max))} stroke="var(--fg)" strokeWidth={4} strokeLinecap="round" fill="none" />
      <line x1={tx1} y1={ty1} x2={tx2} y2={ty2} stroke="var(--accent)" strokeWidth={2} />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="var(--fg)" strokeWidth={2} strokeLinecap="round" />
      <circle cx={cx} cy={cy} r={4} fill="var(--fg)" />
      <text x={pt(0)[0]} y={cy + 14} fontSize="10" fill="var(--chart-axis)" textAnchor="middle">0%</text>
      <text x={pt(max)[0]} y={cy + 14} fontSize="10" fill="var(--chart-axis)" textAnchor="middle">{pct(max, 0)}</text>
    </svg>
  );
}

/** Waterfall: base → each top reason → other inputs → the model's output, in the explanation's units. */
export function ReasonsWaterfall({ r }: { r: ScoreResponse }) {
  const reasons = r.top_reasons;
  const exact = r.explanation_exact && r.explanation_base_value !== null && r.explanation_base_value !== undefined;
  const target = r.explanation_units.startsWith("log-odds") ? logit(r.raw_score) : r.raw_score;
  type Step = { name: string; start: number; end: number; delta: number; kind: "base" | "up" | "down" | "total" };
  const steps: Step[] = [];
  if (exact) {
    let level = r.explanation_base_value as number;
    steps.push({ name: "Base", start: 0, end: level, delta: level, kind: "base" });
    for (const x of reasons) {
      steps.push({ name: x.feature, start: level, end: level + x.contribution, delta: x.contribution, kind: x.contribution >= 0 ? "up" : "down" });
      level += x.contribution;
    }
    const rest = target - level;
    if (Math.abs(rest) > 1e-6) steps.push({ name: "Other inputs", start: level, end: target, delta: rest, kind: rest >= 0 ? "up" : "down" });
    steps.push({ name: "Model output", start: 0, end: target, delta: target, kind: "total" });
  } else {
    for (const x of reasons) steps.push({ name: x.feature, start: 0, end: x.contribution, delta: x.contribution, kind: x.contribution >= 0 ? "up" : "down" });
  }
  const data = steps.map((s) => ({ ...s, lo: Math.min(s.start, s.end), span: Math.abs(s.end - s.start) }));
  const color = (k: Step["kind"]) => (k === "up" ? "var(--diverging-neg)" : k === "down" ? "var(--diverging-pos)" : "var(--fg-muted)");
  return (
    <ChartCard
      title="Why this score"
      description={`${r.explanation_method}. Units: ${r.explanation_units}. Red raises risk, blue lowers it. Association, not causation.`}
      provenance={["model"]}
      height={Math.max(220, data.length * 26 + 40)}
      table={{ columns: ["Input", "Value", "Contribution"], rows: reasons.map((x) => [x.feature, x.value, x.contribution.toFixed(3)]) }}
    >
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }} barCategoryGap={4}>
          <CartesianGrid {...gridProps} horizontal={false} vertical />
          <XAxis type="number" {...axisProps} tickFormatter={(v: number) => v.toFixed(1)} />
          <YAxis type="category" dataKey="name" {...axisProps} width={150} interval={0} tick={{ fill: "var(--fg-secondary)", fontSize: 11 }} />
          <ReferenceLine x={0} stroke="var(--border-strong)" />
          <RTooltip
            cursor={{ fill: "var(--surface-2)" }}
            content={({ active, payload }) =>
              active && payload?.length ? (
                <ChartTooltip
                  active
                  label={String(payload[0].payload.name)}
                  payload={[{ name: "Contribution", value: Number(payload[0].payload.delta), color: color(payload[0].payload.kind) }]}
                  format={(v) => (Number(v) > 0 ? "+" : "") + Number(v).toFixed(3)}
                />
              ) : null
            }
          />
          <Bar dataKey="lo" stackId="w" fill="transparent" isAnimationActive={false} />
          <Bar dataKey="span" stackId="w" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={color(d.kind)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function ResultPanel({ r, cutoffs }: { r: ScoreResponse | DecideResponse; cutoffs: number[] }) {
  const p = r.provenance as Record<string, string[]>;
  const decide = "rules" in r ? (r as DecideResponse) : null;
  return (
    <div className="flex flex-col gap-4" data-testid="score-result">
      <Card>
        <CardHeader
          title="Assessment"
          description={
            <span className="font-mono">
              request {shortId(r.request_id, 12)} · model v{r.model_version} · policy {r.policy_version}
            </span>
          }
          actions={<OutcomeBadge outcome={r.outcome} />}
        />
        <CardBody className="grid grid-cols-1 items-center gap-6 sm:grid-cols-[auto_1fr]">
          <div className="flex flex-col items-center">
            <PdGauge pd={r.pd} threshold={r.threshold} cutoffs={cutoffs} />
            <div className="-mt-3 text-center">
              <div className="num font-mono text-3xl font-semibold text-fg" data-testid="pd-value">
                {pct(r.pd, 1)}
              </div>
              <div className="mt-1 flex items-center justify-center gap-1.5 text-xs text-fg-muted">
                {r.is_calibrated ? "calibrated PD" : "model score (uncalibrated)"} <Provenance labels={prov(p, "pd")} />
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-4">
            <Stat label="Risk band" provenance={<Provenance labels={prov(p, "risk_band")} />} value={<BandBadge band={r.risk_band} />} />
            <Stat label="Threshold" provenance={<Provenance labels={prov(p, "threshold")} />} value={pct(r.threshold)} sub={r.threshold_source} />
            <Stat label="Expected loss" provenance={<Provenance labels={prov(p, "expected_loss")} />} value={money(r.expected_loss)} sub={`LGD ${pct(r.lgd, 0)} × EAD ${money(r.ead)}`} />
            <Stat label="Raw score" provenance={<Provenance labels={prov(p, "raw_score")} />} value={r.raw_score.toFixed(4)} sub="before calibration" />
          </div>
        </CardBody>
      </Card>

      {decide ? (
        <Card>
          <CardHeader title="Policy rules" description={`${decide.policy_name}: ${decide.rules_fired.length} of ${decide.rules.length} fired`} />
          <ul className="divide-y divide-border">
            {decide.rules.map((rule) => (
              <li key={rule.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[13px]">
                <span
                  className="inline-flex h-5 items-center rounded-full px-2 text-[11px] font-medium text-fg"
                  style={{ background: rule.fired ? (rule.action === "flag" ? "var(--serious-soft)" : "var(--warning-soft)") : "var(--surface-2)" }}
                >
                  {rule.fired ? (rule.action === "flag" ? "fired · flag" : "fired · note") : "not fired"}
                </span>
                <span className="font-mono text-xs text-fg">{rule.id}</span>
                <span className="text-fg-muted">{rule.reason}</span>
                {rule.value !== null && rule.value !== undefined ? (
                  <span className="num ml-auto font-mono text-xs text-fg-secondary">
                    {typeof rule.value === "number" ? rule.value.toFixed(3) : rule.value}
                    {rule.limit !== null && rule.limit !== undefined ? ` / ${Array.isArray(rule.limit) ? rule.limit.join(", ") : rule.limit}` : ""}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
          {decide.reasons.length ? (
            <div className="border-t border-border px-4 py-2.5 text-xs text-fg-secondary">
              <span className="font-medium text-fg">Why flagged: </span>
              {decide.reasons.join(" · ")}
            </div>
          ) : null}
        </Card>
      ) : null}

      <ReasonsWaterfall r={r} />

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader title="Uncertainty" description="How much to trust this estimate (held-out test applicants)" />
          <CardBody className="grid grid-cols-2 gap-4">
            <Stat label="Test ROC-AUC" provenance={<Provenance labels={prov(p, "test_roc_auc")} />} value={r.uncertainty.test_roc_auc?.toFixed(3) ?? "—"} />
            <Stat label="Calibration error" provenance={<Provenance labels={prov(p, "test_calibration_error")} />} value={pct(r.uncertainty.test_calibration_error, 2)} sub="mean |PD − observed|" />
            <Stat
              label="Observed default rate in band" provenance={<Provenance labels={prov(p, "band_observed_default_rate")} />}
              value={pct(r.uncertainty.band_observed_default_rate)}
              sub={`${r.uncertainty.band_n.toLocaleString()} test applicants in ${r.risk_band}`}
            />
            <Stat label="Mean PD in band" provenance={<Provenance labels={prov(p, "band_mean_pd")} />} value={pct(r.uncertainty.band_mean_pd)} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Caveats" />
          <ul className="flex flex-col gap-2 px-4 py-3 text-xs text-fg-secondary">
            {r.warnings.map((w) => (
              <li key={w} className="flex gap-2">
                <AlertTriangle aria-hidden className="mt-0.5 size-3.5 shrink-0" style={{ color: "var(--serious)" }} />
                {w}
              </li>
            ))}
            {r.caveats.map((c) => (
              <li key={c} className="flex gap-2">
                <Info aria-hidden className="mt-0.5 size-3.5 shrink-0 text-fg-muted" />
                {c}
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </div>
  );
}
