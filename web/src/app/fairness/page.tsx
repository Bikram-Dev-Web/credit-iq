"use client";

import * as React from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { Info } from "lucide-react";
import { ChartCard, ChartTooltip, Legend, SERIES, axisProps, gridProps } from "@/components/charts/chart";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Badge, Card, CardHeader, ErrorState, Field, Select } from "@/components/ui/primitives";
import { useFairness } from "@/lib/api/hooks";
import { dec, num, pct } from "@/lib/format";

type Row = Record<string, string | number | boolean | null>;
type Report = { attribute: string; available_attributes: string[]; policy_version: string; groups: Row[]; disparities: Row[]; proxy_scan: Row[]; missingness: Row[]; guidance: string[]; min_group_size: number };

const groupCols: ColumnDef<Row, unknown>[] = [
  { accessorKey: "group", header: "Group", cell: (c) => <span className="flex items-center gap-1.5">{String(c.getValue())}{c.row.original.small_sample ? <Badge tone="warning">small</Badge> : null}</span> },
  { accessorKey: "n", header: "Applicants", meta: { numeric: true }, cell: (c) => num(c.getValue()) },
  { accessorKey: "observed_default_rate", header: "Observed DR", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "mean_pd", header: "Mean PD", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "within_threshold_rate", header: "Within threshold", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "tpr_recall", header: "Defaulters flagged", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "fpr_good_flagged", header: "Good flagged", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "calibration_gap", header: "Calibration gap", meta: { numeric: true }, cell: (c) => pct(c.getValue(), 2) },
  { accessorKey: "roc_auc", header: "ROC-AUC", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
];
const dispCols: ColumnDef<Row, unknown>[] = [
  { accessorKey: "metric", header: "Metric" },
  { accessorKey: "highest_group", header: "Highest" },
  { accessorKey: "lowest_group", header: "Lowest" },
  { accessorKey: "difference", header: "Gap", meta: { numeric: true }, cell: (c) => pct(c.getValue(), 1) },
  { accessorKey: "ratio_low_to_high", header: "Ratio", meta: { numeric: true }, cell: (c) => dec(c.getValue(), 2) },
  { accessorKey: "what_it_checks", header: "What it checks", cell: (c) => <span className="whitespace-normal text-xs text-fg-muted">{String(c.getValue())}</span> },
];
const proxyCols: ColumnDef<Row, unknown>[] = [
  { accessorKey: "feature", header: "Model input" },
  { accessorKey: "association", header: "Association", meta: { numeric: true }, cell: (c) => dec(c.getValue(), 3) },
  { accessorKey: "level", header: "Level", cell: (c) => (c.getValue() ? <Badge tone={c.getValue() === "strong" ? "critical" : c.getValue() === "moderate" ? "warning" : "neutral"}>{String(c.getValue())}</Badge> : "—") },
  { accessorKey: "method", header: "Method", cell: (c) => <span className="text-xs text-fg-muted">{String(c.getValue())}</span> },
];

export default function FairnessPage() {
  const [attribute, setAttribute] = React.useState<string | undefined>(undefined);
  const [split, setSplit] = React.useState<"test" | "validation">("test");
  const { data, isLoading, error } = useFairness({ attribute, split });
  const r = data?.result as Report | undefined;
  const p = data?.provenance as Record<string, string[]> | undefined;
  const chartRows = (r?.groups ?? []).filter((g) => !g.small_sample);
  return (
    <>
      <PageHeader
        title="Fairness"
        description="Group-level audit by a sensitive attribute that is not a model input. Uses the active policy's thresholds on held-out applicants."
        actions={
          <>
            <Field label="Attribute" className="w-48"><Select ariaLabel="Sensitive attribute" value={attribute ?? r?.attribute} onValueChange={setAttribute} options={(r?.available_attributes ?? []).map((a) => ({ value: a, label: a }))} /></Field>
            <Field label="Split" className="w-36"><Select ariaLabel="Split" value={split} onValueChange={(v) => setSplit(v as "test" | "validation")} options={[{ value: "test", label: "Test" }, { value: "validation", label: "Validation" }]} /></Field>
          </>
        }
      />
      {error ? <ErrorState error={error} className="mb-4" /> : null}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Flag rate and defaulters flagged, by group"
          description={`Groups below ${r?.min_group_size ?? 30} applicants or 10 defaults are omitted (too noisy)`}
          provenance={prov(p, "flag_rate")}
          loading={isLoading}
          legend={<Legend items={[{ label: "Flag rate", color: SERIES[0] }, { label: "Defaulters flagged (recall)", color: SERIES[1] }]} />}
          table={{ columns: ["Group", "Flag rate", "Recall", "n"], rows: chartRows.map((g) => [String(g.group), pct(g.flag_rate), pct(g.tpr_recall), num(g.n)]) }}
        >
          <ResponsiveContainer>
            <BarChart data={chartRows} margin={{ top: 4, right: 8, bottom: 0, left: -8 }} barGap={2}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="group" {...axisProps} interval={0} />
              <YAxis {...axisProps} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => pct(Number(v))} />} />
              <Bar dataKey="flag_rate" name="Flag rate" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
              <Bar dataKey="tpr_recall" name="Defaulters flagged" fill={SERIES[1]} radius={[4, 4, 0, 0]} maxBarSize={28} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
        <Card>
          <CardHeader title="Responsible use" description="Read before drawing conclusions" />
          <ul className="flex flex-col gap-2 px-4 py-3 text-xs text-fg-secondary">
            {(r?.guidance ?? []).map((g) => (
              <li key={g} className="flex gap-2"><Info aria-hidden className="mt-0.5 size-3.5 shrink-0 text-accent" />{g}</li>
            ))}
          </ul>
        </Card>
      </div>
      <Card className="mt-4">
        <CardHeader title={<span className="flex items-center gap-2">Group metrics {r ? <Badge className="font-mono">{r.attribute}</Badge> : null}</span>} description={<Provenance labels={prov(p, "within_threshold_rate")} />} />
        <DataTable columns={groupCols} data={r?.groups as Row[] | undefined} loading={isLoading} csvName="creditiq-fairness-groups" />
      </Card>
      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader title="Disparity summary" description="Most vs least favourable group (small groups excluded)" />
          <DataTable columns={dispCols} data={r?.disparities as Row[] | undefined} loading={isLoading} csvName="creditiq-disparities" emptyBody="Fewer than two groups are large enough to compare." />
        </Card>
        <Card>
          <CardHeader title="Proxy scan" description="Association between each model input and the groups — possible proxies" />
          <DataTable columns={proxyCols} data={r?.proxy_scan as Row[] | undefined} loading={isLoading} csvName="creditiq-proxy-scan" pageSize={10} initialSorting={[{ id: "association", desc: true }]} />
        </Card>
      </div>
    </>
  );
}
