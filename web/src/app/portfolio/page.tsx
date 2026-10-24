"use client";

import * as React from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { ChartCard, ChartTooltip, Legend, SERIES, axisProps, gridProps } from "@/components/charts/chart";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { BAND_VAR, KpiCard, PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Card, CardHeader, ErrorState, Field, Input, Select } from "@/components/ui/primitives";
import { usePortfolio } from "@/lib/api/hooks";
import { money, num, pct } from "@/lib/format";

type Row = Record<string, number | string | null>;
type Split = "validation" | "test";

function useDebounced<T>(v: T, ms = 350) {
  const [d, setD] = React.useState(v);
  React.useEffect(() => {
    const t = setTimeout(() => setD(v), ms);
    return () => clearTimeout(t);
  }, [v, ms]);
  return d;
}

const segColumns: ColumnDef<Row, unknown>[] = [
  { accessorKey: "segment", header: "Segment" },
  { accessorKey: "loans", header: "Loans", meta: { numeric: true }, cell: (c) => num(c.getValue()) },
  { accessorKey: "ead_share", header: "EAD share", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "observed_default_rate", header: "Observed DR", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "avg_pd", header: "Avg PD", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "expected_loss", header: "Exp. loss", meta: { numeric: true }, cell: (c) => money(c.getValue()) },
  { accessorKey: "el_rate", header: "EL rate", meta: { numeric: true }, cell: (c) => pct(c.getValue(), 2) },
  { accessorKey: "el_share", header: "EL share", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
];

export default function PortfolioPage() {
  const [split, setSplit] = React.useState<Split>("test");
  const [lgdText, setLgdText] = React.useState("45");
  const [ccfText, setCcfText] = React.useState("100");
  const [column, setColumn] = React.useState<string | undefined>(undefined);
  const lgdN = Number(lgdText) / 100;
  const ccfN = Number(ccfText) / 100;
  const lgdValid = Number.isFinite(lgdN) && lgdN >= 0 && lgdN <= 1;
  const ccfValid = Number.isFinite(ccfN) && ccfN > 0 && ccfN <= 2;
  const q = useDebounced({ split, lgd: lgdValid ? lgdN : undefined, ccf: ccfValid ? ccfN : undefined });

  const summary = usePortfolio("summary", q);
  const bands = usePortfolio("bands", q);
  const deciles = usePortfolio("deciles", q);
  const segments = usePortfolio("segments", { ...q, column });
  const conc = usePortfolio("concentration", { ...q, column });
  const s = summary.data?.summary as Record<string, number> | undefined;
  const p = summary.data?.provenance as Record<string, string[]> | undefined;
  const segSummary = segments.data?.summary as { column?: string; available_columns?: string[]; hhi?: number; effective_segments?: number; top3_share?: number } | undefined;
  const bandRows = (bands.data?.rows ?? []) as Row[];

  return (
    <>
      <PageHeader
        title="Portfolio"
        description="Held-out applicants treated as a portfolio. Observed default rate (labels), average PD (model) and expected loss (model × your LGD/CCF assumptions) are kept separate."
      />
      <Card className="mb-4">
        <div className="flex flex-wrap items-end gap-4 px-4 py-3">
          <Field label="Split" className="w-40">
            <Select value={split} ariaLabel="Split" onValueChange={(v) => setSplit(v as Split)} options={[{ value: "test", label: "Test (default)" }, { value: "validation", label: "Validation" }]} />
          </Field>
          <Field label={<>LGD % <Provenance labels={["assumption"]} /></>} htmlFor="lgd" error={lgdValid ? undefined : "0–100"} className="w-28">
            <Input id="lgd" inputMode="decimal" value={lgdText} onChange={(e) => setLgdText(e.target.value)} className="num font-mono" aria-invalid={!lgdValid} />
          </Field>
          <Field label={<>CCF % <Provenance labels={["assumption"]} /></>} htmlFor="ccf" error={ccfValid ? undefined : "1–200"} className="w-28">
            <Input id="ccf" inputMode="decimal" value={ccfText} onChange={(e) => setCcfText(e.target.value)} className="num font-mono" aria-invalid={!ccfValid} />
          </Field>
          <p className="max-w-md text-xs text-fg-muted">EAD = exposure × CCF; EL = PD × LGD × EAD (12-month). Missing exposures use the policy&apos;s default.</p>
        </div>
      </Card>
      {summary.error ? <ErrorState error={summary.error} className="mb-4" /> : null}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <KpiCard label="Loans" loading={summary.isLoading} value={num(s?.n_loans)} provenance={prov(p, "n_loans")} />
        <KpiCard label="Total EAD" loading={summary.isLoading} value={money(s?.total_ead)} provenance={prov(p, "total_ead")} />
        <KpiCard label="12-month expected loss" loading={summary.isLoading} value={money(s?.total_el)} provenance={prov(p, "total_el")} sub={`EL rate ${pct(s?.el_rate, 2)}`} />
        <KpiCard label="Observed default rate" loading={summary.isLoading} value={pct(s?.observed_default_rate)} provenance={prov(p, "observed_default_rate")} sub={`average PD ${pct(s?.avg_pd)}`} />
        <KpiCard label="High-risk share (EAD)" loading={summary.isLoading} value={pct(s?.high_risk_share_ead)} provenance={prov(p, "high_risk_share_ead")} sub="Elevated + High bands" />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Exposure by risk band"
          provenance={prov(p, "ead")}
          loading={bands.isLoading}
          table={{ columns: ["Band", "Loans", "EAD", "EAD share"], rows: bandRows.map((r) => [String(r.risk_band), num(r.loans), money(r.ead), pct(r.ead_share)]) }}
        >
          <ResponsiveContainer>
            <BarChart data={bandRows} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="risk_band" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} />} />
              <Bar dataKey="ead" name="EAD" radius={[4, 4, 0, 0]} maxBarSize={48}>
                {bandRows.map((r) => (
                  <Cell key={String(r.risk_band)} fill={BAND_VAR[String(r.risk_band)]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard
          title="Expected loss by risk band"
          description="Same bands; EL concentrates where PD is high"
          provenance={prov(p, "expected_loss")}
          loading={bands.isLoading}
          table={{ columns: ["Band", "Expected loss", "EL rate", "EL share"], rows: bandRows.map((r) => [String(r.risk_band), money(r.expected_loss), pct(r.el_rate, 2), pct(r.el_share)]) }}
        >
          <ResponsiveContainer>
            <BarChart data={bandRows} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="risk_band" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} />} />
              <Bar dataKey="expected_loss" name="Expected loss" radius={[4, 4, 0, 0]} maxBarSize={48}>
                {bandRows.map((r) => (
                  <Cell key={String(r.risk_band)} fill={BAND_VAR[String(r.risk_band)]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard
          title="Observed vs predicted by PD decile"
          description="1 = lowest-risk decile. Close lines mean well-calibrated PDs."
          provenance={["observed", "model"]}
          loading={deciles.isLoading}
          legend={<Legend items={[{ label: "Observed default rate", color: SERIES[1] }, { label: "Average PD", color: SERIES[0] }]} />}
          table={{
            columns: ["Decile", "Loans", "Observed DR", "Average PD"],
            rows: ((deciles.data?.rows ?? []) as Row[]).map((r) => [num(r.decile), num(r.loans), pct(r.observed_default_rate), pct(r.avg_pd)]),
          }}
        >
          <ResponsiveContainer>
            <LineChart data={(deciles.data?.rows ?? []) as Row[]} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="decile" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <RTooltip content={<ChartTooltip format={(v) => pct(Number(v))} labelFormat={(l) => `Decile ${l}`} />} />
              <Line type="monotone" dataKey="observed_default_rate" name="Observed default rate" stroke={SERIES[1]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="avg_pd" name="Average PD" stroke={SERIES[0]} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
        <ChartCard
          title="Loss concentration"
          description={
            conc.data?.summary && (conc.data.summary as Record<string, number>).el_share_riskiest_10pct !== undefined
              ? `The riskiest 10% of loans carry ${pct((conc.data.summary as Record<string, number>).el_share_riskiest_10pct)} of expected loss`
              : "Share of expected loss vs share of loans (riskiest first)"
          }
          provenance={prov(conc.data?.provenance as Record<string, string[]>, "share_of_expected_loss")}
          loading={conc.isLoading}
          table={{ columns: ["Share of loans", "Share of EL"], rows: ((conc.data?.rows ?? []) as Row[]).filter((_, i) => i % 5 === 0).map((r) => [pct(r.share_of_loans, 0), pct(r.share_of_expected_loss)]) }}
        >
          <ResponsiveContainer>
            <AreaChart data={(conc.data?.rows ?? []) as Row[]} margin={{ top: 4, right: 8, bottom: 0, left: -8 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="share_of_loans" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <YAxis {...axisProps} domain={[0, 1]} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <RTooltip content={<ChartTooltip format={(v) => pct(Number(v))} labelFormat={(l) => `${pct(Number(l), 0)} of loans`} />} />
              <Area type="monotone" dataKey="share_of_expected_loss" name="Share of expected loss" stroke={SERIES[0]} strokeWidth={2} fill={SERIES[0]} fillOpacity={0.12} isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <Card className="mt-4">
        <CardHeader
          title="Segment concentration"
          description={
            segSummary?.hhi !== undefined ? (
              <span className="num">
                HHI {segSummary.hhi?.toFixed(3)} · effective segments {segSummary.effective_segments?.toFixed(1)} · top-3 share {pct(segSummary.top3_share)}{" "}
                <Provenance labels={prov(segments.data?.provenance as Record<string, string[]>, "hhi")} />
              </span>
            ) : (
              "Exposure and EL by segment"
            )
          }
          actions={
            <div className="w-56">
              <Select
                ariaLabel="Segment column"
                value={column ?? segSummary?.column}
                onValueChange={setColumn}
                options={(segSummary?.available_columns ?? []).map((c) => ({ value: c, label: c }))}
              />
            </div>
          }
        />
        <DataTable columns={segColumns} data={(segments.data?.rows ?? []) as Row[]} loading={segments.isLoading} csvName="creditiq-segments" pageSize={15} />
      </Card>
    </>
  );
}
