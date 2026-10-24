"use client";

import * as React from "react";
import { useMutation } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { Plus, Trash2 } from "lucide-react";
import { ChartCard, ChartTooltip, SERIES, axisProps, gridProps } from "@/components/charts/chart";
import { KpiCard, PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Button, Card, CardBody, CardHeader, ErrorState, Field, Input, Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { money, num, pct } from "@/lib/format";

type Num = Record<string, number>;

function NumInput({ label, value, onChange, pctMode, provenance = ["assumption"] }: { label: string; value: number; onChange: (v: number) => void; pctMode?: boolean; provenance?: string[] }) {
  const show = (v: number) => String(pctMode ? Number((v * 100).toPrecision(6)) : v);
  const [text, setText] = React.useState(show(value));
  const [seen, setSeen] = React.useState(value);
  if (value !== seen) {
    setSeen(value);
    setText(show(value));
  }
  return (
    <Field label={<>{label}{pctMode ? " %" : ""} <Provenance labels={provenance} /></>}>
      <Input
        inputMode="decimal"
        value={text}
        className="num font-mono"
        onChange={(e) => {
          setText(e.target.value);
          const n = Number(e.target.value);
          if (e.target.value.trim() !== "" && Number.isFinite(n)) onChange(pctMode ? n / 100 : n);
        }}
      />
    </Field>
  );
}

function useAuto<T>(fn: (body: T) => Promise<unknown>, body: T) {
  const m = useMutation({ mutationFn: fn });
  const key = JSON.stringify(body);
  React.useEffect(() => {
    const t = setTimeout(() => m.mutate(body), 350);
    return () => clearTimeout(t);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return m;
}

/* ------------------------------------------------------------------ Economics */
function Economics() {
  const [a, setA] = React.useState({ n_loans: 1000, avg_exposure: 500000, pd: 0.08, lgd: 0.45, annual_rate: 0.16, funding_rate: 0.07, opex_per_loan: 1500, horizon_years: 1 });
  const set = (k: keyof typeof a) => (v: number) => setA((s) => ({ ...s, [k]: v }));
  const m = useAuto((body: typeof a) => unwrap(api.POST("/v1/scenarios/economics", { body })), a);
  const out = m.data as { result: { contribution: Num; sensitivity: Num[]; heatmap: { pd: number; annual_rate: number; contribution_margin_on_exposure: number }[] }; provenance: Record<string, string[]> } | undefined;
  const c = out?.result.contribution;
  const p = out?.provenance;
  const water = c
    ? [
        { name: "Interest", v: c.expected_interest_income },
        { name: "Fees", v: c.fee_income },
        { name: "Funding", v: -c.funding_cost },
        { name: "Credit loss", v: -c.expected_credit_loss },
        { name: "Operating", v: -c.operating_cost },
      ]
    : [];
  let level = 0;
  const steps = water.map((w) => {
    const start = level;
    level += w.v;
    return { ...w, lo: Math.min(start, level), span: Math.abs(w.v) };
  });
  if (c) steps.push({ name: "Contribution", v: c.expected_contribution, lo: Math.min(0, c.expected_contribution), span: Math.abs(c.expected_contribution) });
  const pds = Array.from(new Set(out?.result.heatmap.map((h) => h.pd) ?? []));
  const rates = Array.from(new Set(out?.result.heatmap.map((h) => h.annual_rate) ?? []));
  const cell = (pdv: number, r: number) => out?.result.heatmap.find((h) => h.pd === pdv && h.annual_rate === r)?.contribution_margin_on_exposure ?? 0;
  const maxAbs = Math.max(0.001, ...(out?.result.heatmap.map((h) => Math.abs(h.contribution_margin_on_exposure)) ?? [0]));
  const heat = (v: number) => {
    const t = Math.min(1, Math.abs(v) / maxAbs);
    return `color-mix(in oklab, var(${v >= 0 ? "--diverging-pos" : "--diverging-neg"}) ${Math.round(t * 85)}%, var(--diverging-mid))`;
  };
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[20rem_minmax(0,1fr)]">
      <Card className="self-start">
        <CardHeader title="Assumptions" description="Simplified one-period contribution; no taxes, capital or discounting" />
        <CardBody className="grid grid-cols-2 gap-3">
          <NumInput label="Loans" value={a.n_loans} onChange={set("n_loans")} />
          <NumInput label="Avg exposure" value={a.avg_exposure} onChange={set("avg_exposure")} />
          <NumInput label="Annual PD" value={a.pd} onChange={set("pd")} pctMode />
          <NumInput label="LGD" value={a.lgd} onChange={set("lgd")} pctMode />
          <NumInput label="Lending rate" value={a.annual_rate} onChange={set("annual_rate")} pctMode />
          <NumInput label="Funding rate" value={a.funding_rate} onChange={set("funding_rate")} pctMode />
          <NumInput label="Opex / loan / yr" value={a.opex_per_loan} onChange={set("opex_per_loan")} />
          <NumInput label="Horizon (years)" value={a.horizon_years} onChange={set("horizon_years")} />
        </CardBody>
      </Card>
      <div className="flex min-w-0 flex-col gap-4">
        {m.error ? <ErrorState error={m.error} /> : null}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <KpiCard label="Expected contribution" value={money(c?.expected_contribution)} provenance={prov(p, "expected_contribution")} sub={`margin on exposure ${pct(c?.contribution_margin_on_exposure, 2)}`} loading={!c} />
          <KpiCard label="Break-even annual PD" value={pct(c?.break_even_pd, 2)} provenance={prov(p, "break_even_pd")} sub="contribution = 0" loading={!c} />
          <KpiCard label="Break-even lending rate" value={pct(c?.break_even_rate, 2)} provenance={prov(p, "break_even_rate")} sub="risk-based pricing floor" loading={!c} />
        </div>
        <ChartCard title="Lending economics" description="From interest income to expected contribution" provenance={prov(p, "expected_contribution")} loading={!c} table={{ columns: ["Component", "Amount"], rows: steps.map((s) => [s.name, money(s.v)]) }}>
          <ResponsiveContainer>
            <BarChart data={steps} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="name" {...axisProps} interval={0} />
              <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
              <ReferenceLine y={0} stroke="var(--border-strong)" />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={({ active, payload }) => (active && payload?.length ? <ChartTooltip active label={String(payload[0].payload.name)} payload={[{ name: "Amount", value: Number(payload[0].payload.v), color: Number(payload[0].payload.v) >= 0 ? SERIES[0] : "var(--diverging-neg)" }]} format={(v) => money(Number(v))} /> : null)} />
              <Bar dataKey="lo" stackId="w" fill="transparent" isAnimationActive={false} />
              <Bar dataKey="span" stackId="w" radius={[4, 4, 4, 4]} isAnimationActive={false}>
                {steps.map((s) => (
                  <Cell key={s.name} fill={s.name === "Contribution" ? "var(--accent)" : s.v >= 0 ? SERIES[0] : "var(--diverging-neg)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <Card>
          <CardHeader title="Contribution margin: PD × lending rate" description="Blue = profitable, red = loss-making, grey ≈ break-even. Hover a cell for the value." />
          <CardBody className="overflow-x-auto">
            {pds.length ? (
              <table className="text-[11px]" role="grid" aria-label="Contribution margin heatmap">
                <thead>
                  <tr>
                    <th className="px-1 py-1 text-left font-medium text-fg-muted">PD ↓ / rate →</th>
                    {rates.map((r) => (
                      <th key={r} className="num px-1 py-1 font-mono font-normal text-fg-muted">{pct(r, 0)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pds.map((pdv) => (
                    <tr key={pdv}>
                      <th className="num px-1 py-0.5 text-left font-mono font-normal text-fg-muted">{pct(pdv, 1)}</th>
                      {rates.map((r) => {
                        const v = cell(pdv, r);
                        return (
                          <td key={r} className="p-0.5">
                            <div title={`PD ${pct(pdv, 1)}, rate ${pct(r, 0)}: margin ${pct(v, 2)}`} className="num flex h-7 w-14 items-center justify-center rounded-[3px] font-mono text-[10px] text-fg" style={{ background: heat(v) }}>
                              {pct(v, 1)}
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="skeleton h-64" />
            )}
          </CardBody>
        </Card>
        <ChartCard title="Sensitivity (±20%)" description="Change in contribution when one driver moves" provenance={prov(p, "range")} loading={!out} table={{ columns: ["Driver", "Down", "Up"], rows: (out?.result.sensitivity ?? []).map((s) => [String(s.driver), money(s.change_if_driver_down), money(s.change_if_driver_up)]) }}>
          <ResponsiveContainer>
            <BarChart data={[...(out?.result.sensitivity ?? [])].reverse()} layout="vertical" margin={{ top: 0, right: 12, bottom: 0, left: 8 }} stackOffset="sign">
              <CartesianGrid {...gridProps} horizontal={false} vertical />
              <XAxis type="number" {...axisProps} tickFormatter={(v: number) => money(v)} />
              <YAxis type="category" dataKey="driver" {...axisProps} width={150} tick={{ fill: "var(--fg-secondary)", fontSize: 11 }} />
              <ReferenceLine x={0} stroke="var(--border-strong)" />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} />} />
              <Bar dataKey="change_if_driver_down" name="Driver −20%" fill={SERIES[0]} radius={[4, 4, 4, 4]} isAnimationActive={false} />
              <Bar dataKey="change_if_driver_up" name="Driver +20%" fill={SERIES[1]} radius={[4, 4, 4, 4]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Stress */
type Scenario = { scenario: string; pd_multiplier: number; lgd_shift: number; ead_multiplier: number };
const DEFAULT_SCENARIOS: Scenario[] = [
  { scenario: "Improved", pd_multiplier: 0.8, lgd_shift: -0.05, ead_multiplier: 1 },
  { scenario: "Base", pd_multiplier: 1, lgd_shift: 0, ead_multiplier: 1 },
  { scenario: "Adverse", pd_multiplier: 1.5, lgd_shift: 0.1, ead_multiplier: 1.05 },
  { scenario: "Severe stress", pd_multiplier: 2.2, lgd_shift: 0.2, ead_multiplier: 1.1 },
];

function Stress() {
  const [rows, setRows] = React.useState<Scenario[]>(DEFAULT_SCENARIOS);
  const [lgd, setLgd] = React.useState(0.45);
  const m = useAuto((body: { scenarios: Scenario[]; lgd: number }) => unwrap(api.POST("/v1/scenarios/stress", { body })), { scenarios: rows, lgd });
  const out = m.data as { result: { scenarios: Num[] & { scenario: string }[] }; provenance: Record<string, string[]> } | undefined;
  const res = (out?.result.scenarios ?? []) as unknown as (Num & { scenario: string })[];
  const setCell = (i: number, k: keyof Scenario, v: string) =>
    setRows((r) => r.map((x, j) => (j === i ? { ...x, [k]: k === "scenario" ? v : Number(v) } : x)));
  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader title="Scenarios" description="Per-loan PD multipliers, LGD shifts and EAD multipliers on the held-out portfolio (PD and LGD capped at 100%)" actions={<NumInput label="Base LGD" value={lgd} onChange={setLgd} pctMode />} />
        <div className="overflow-x-auto">
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-border text-[11px] uppercase tracking-wide text-fg-muted">
                <th className="px-4 py-2 text-left font-medium">Scenario</th>
                <th className="px-2 py-2 text-right font-medium">PD ×</th>
                <th className="px-2 py-2 text-right font-medium">LGD shift</th>
                <th className="px-2 py-2 text-right font-medium">EAD ×</th>
                <th className="px-4 py-2 text-right font-medium">Expected loss</th>
                <th className="px-4 py-2 text-right font-medium">EL rate</th>
                <th className="px-4 py-2 text-right font-medium">Contribution</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const o = res[i];
                return (
                  <tr key={i} className="border-b border-border/60">
                    <td className="px-4 py-1.5"><Input aria-label="Scenario name" value={r.scenario} onChange={(e) => setCell(i, "scenario", e.target.value)} className="h-7 w-40" /></td>
                    {(["pd_multiplier", "lgd_shift", "ead_multiplier"] as const).map((k) => (
                      <td key={k} className="px-2 py-1.5"><Input aria-label={k} inputMode="decimal" defaultValue={r[k]} onBlur={(e) => setCell(i, k, e.target.value)} className="num ml-auto h-7 w-20 text-right font-mono" /></td>
                    ))}
                    <td className="num px-4 py-1.5 text-right font-mono text-xs">{money(o?.expected_loss)}</td>
                    <td className="num px-4 py-1.5 text-right font-mono text-xs">{pct(o?.el_rate, 2)}</td>
                    <td className="num px-4 py-1.5 text-right font-mono text-xs">{money(o?.expected_contribution)}</td>
                    <td className="px-2"><Button size="icon" variant="ghost" aria-label="Remove scenario" onClick={() => setRows((x) => x.filter((_, j) => j !== i))}><Trash2 /></Button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="px-4 py-2.5">
          <Button size="sm" variant="ghost" disabled={rows.length >= 20} onClick={() => setRows((r) => [...r, { scenario: `Scenario ${r.length + 1}`, pd_multiplier: 1.2, lgd_shift: 0.05, ead_multiplier: 1 }])}><Plus /> Add scenario</Button>
        </div>
      </Card>
      {m.error ? <ErrorState error={m.error} /> : null}
      <ChartCard title="Expected loss by scenario" provenance={prov(out?.provenance, "expected_loss")} loading={!out} table={{ columns: ["Scenario", "Expected loss", "Loans"], rows: res.map((r) => [r.scenario, money(r.expected_loss), num(r.n_loans)]) }}>
        <ResponsiveContainer>
          <BarChart data={res} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
            <CartesianGrid {...gridProps} />
            <XAxis dataKey="scenario" {...axisProps} />
            <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
            <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} />} />
            <Bar dataKey="expected_loss" name="Expected loss" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}

/* ------------------------------------------------------------------ NPV */
function NPV() {
  const [a, setA] = React.useState({ principal: 500000, annual_rate: 0.16, term_months: 36, annual_pd: 0.08, lgd: 0.45, discount_rate: 0.07, recovery_lag_months: 6 });
  const set = (k: keyof typeof a) => (v: number) => setA((s) => ({ ...s, [k]: v }));
  const m = useAuto((body: typeof a) => unwrap(api.POST("/v1/scenarios/npv", { body })), a);
  const out = m.data as { result: Num & { cash_flows: Num[] }; provenance: Record<string, string[]> } | undefined;
  const r = out?.result;
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[20rem_minmax(0,1fr)]">
      <Card className="self-start">
        <CardHeader title="Loan" description="Annuity repayments, constant monthly default hazard, recovery after a lag" />
        <CardBody className="grid grid-cols-2 gap-3">
          <NumInput label="Principal" value={a.principal} onChange={set("principal")} />
          <NumInput label="Term (months)" value={a.term_months} onChange={set("term_months")} />
          <NumInput label="Lending rate" value={a.annual_rate} onChange={set("annual_rate")} pctMode />
          <NumInput label="Annual PD" value={a.annual_pd} onChange={set("annual_pd")} pctMode />
          <NumInput label="LGD" value={a.lgd} onChange={set("lgd")} pctMode />
          <NumInput label="Discount rate" value={a.discount_rate} onChange={set("discount_rate")} pctMode />
          <NumInput label="Recovery lag (m)" value={a.recovery_lag_months} onChange={set("recovery_lag_months")} />
        </CardBody>
      </Card>
      <div className="flex min-w-0 flex-col gap-4">
        {m.error ? <ErrorState error={m.error} /> : null}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <KpiCard label="Risk-adjusted NPV" value={money(r?.npv_risk_adjusted)} provenance={prov(out?.provenance, "npv_risk_adjusted")} loading={!r} />
          <KpiCard label="Contractual NPV" value={money(r?.npv_contractual)} provenance={prov(out?.provenance, "npv_contractual")} loading={!r} />
          <KpiCard label="PV of credit cost" value={money(r?.pv_credit_cost)} provenance={prov(out?.provenance, "pv_credit_cost")} loading={!r} />
          <KpiCard label="Lifetime PD" value={pct(r?.lifetime_pd)} provenance={prov(out?.provenance, "lifetime_pd")} sub={`expected yield ${pct(r?.expected_annual_yield, 2)}`} loading={!r} />
        </div>
        <ChartCard title="Expected cash flows" description="Month 0 is the disbursement; recoveries arrive after the lag" provenance={prov(out?.provenance, "expected_cash_flow")} loading={!r} height={280} table={{ columns: ["Month", "Expected cash flow", "PV"], rows: (r?.cash_flows ?? []).map((c) => [c.month, money(c.expected_cash_flow), money(c.pv_expected)]) }}>
          <ResponsiveContainer>
            <BarChart data={(r?.cash_flows ?? []).slice(1)} margin={{ top: 8, right: 8, bottom: 0, left: 4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="month" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} labelFormat={(l) => `Month ${l}`} />} />
              <Bar dataKey="expected_payments" name="Expected payments" stackId="c" fill={SERIES[0]} isAnimationActive={false} />
              <Bar dataKey="expected_recoveries" name="Expected recoveries" stackId="c" fill={SERIES[2]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}

export default function ScenariosPage() {
  return (
    <>
      <PageHeader title="Scenarios" description="Lending economics, stress tests and loan NPV. Every figure here is simulated from model outputs and your assumptions." />
      <Tabs defaultValue="economics">
        <TabsList className="mb-4">
          <TabsTrigger value="economics">Economics</TabsTrigger>
          <TabsTrigger value="stress">Stress testing</TabsTrigger>
          <TabsTrigger value="npv">Loan NPV</TabsTrigger>
        </TabsList>
        <TabsContent value="economics"><Economics /></TabsContent>
        <TabsContent value="stress"><Stress /></TabsContent>
        <TabsContent value="npv"><NPV /></TabsContent>
      </Tabs>
    </>
  );
}
