"use client";

import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { ExternalLink, Upload } from "lucide-react";
import { toast } from "sonner";
import { ChartCard, ChartTooltip, Legend, SERIES, axisProps, gridProps } from "@/components/charts/chart";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { KpiCard, PageHeader, StatusBadge, prov } from "@/components/ui/domain";
import { Button, Card, CardHeader, EmptyState, ErrorState } from "@/components/ui/primitives";
import { useAlerts, useDrift, useDriftHistory, useMeta } from "@/lib/api/hooks";
import { dateTime, dec, num, pct, pp } from "@/lib/format";

type Feature = { column: string; kind: string; psi: number | null; status: string; ref_missing_rate: number; new_missing_rate: number | null; missing_change: number | null };
type Alert = { id: number; created_at: string; code: string; severity: string; message: string; value: number | null; threshold: number | null; model_version?: string; batch_name?: string };
type Run = {
  id: number;
  created_at: string;
  source: string;
  model_version: string;
  batch_name: string;
  n_rows: number;
  status: string;
  max_feature_psi: number | null;
  n_significant: number;
  score_psi: number | null;
  mean_pd: number | null;
  ref_mean_pd: number | null;
  labels_available: boolean;
  observed_default_rate: number | null;
  roc_auc: number | null;
  alerts: Alert[];
  features?: Feature[];
  score?: { ref_props: number[]; new_props: number[]; cuts: number[] } | null;
};

const featCols: ColumnDef<Feature, unknown>[] = [
  { accessorKey: "column", header: "Input" },
  { accessorKey: "psi", header: "PSI", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
  { accessorKey: "status", header: "Status", cell: (c) => <StatusBadge status={c.getValue() === "Significant shift" ? "critical" : c.getValue() === "Moderate shift" ? "warning" : "ok"} /> },
  { accessorKey: "ref_missing_rate", header: "Missing (train)", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "new_missing_rate", header: "Missing (batch)", meta: { numeric: true }, cell: (c) => pct(c.getValue()) },
  { accessorKey: "missing_change", header: "Change", meta: { numeric: true }, cell: (c) => pp(c.getValue()) },
];
const alertCols: ColumnDef<Alert, unknown>[] = [
  { accessorKey: "created_at", header: "Time", cell: (c) => <span className="font-mono text-xs text-fg-secondary">{dateTime(c.getValue())}</span> },
  { accessorKey: "severity", header: "Severity", cell: (c) => <StatusBadge status={String(c.getValue())} /> },
  { accessorKey: "code", header: "Code", cell: (c) => <span className="font-mono text-xs">{String(c.getValue())}</span> },
  { accessorKey: "message", header: "Message", cell: (c) => <span className="whitespace-normal">{String(c.getValue())}</span> },
  { accessorKey: "batch_name", header: "Batch" },
  { accessorKey: "model_version", header: "Model", cell: (c) => <span className="font-mono text-xs">v{String(c.getValue())}</span> },
];

export default function MonitoringPage() {
  const qc = useQueryClient();
  const meta = useMeta();
  const drift = useDrift();
  const history = useDriftHistory();
  const alerts = useAlerts();
  const fileRef = React.useRef<HTMLInputElement>(null);
  const [busy, setBusy] = React.useState(false);
  const run = drift.data?.result as Run | null | undefined;
  const p = drift.data?.provenance as Record<string, string[]> | undefined;
  const features = [...(run?.features ?? [])].filter((f) => f.psi !== null).sort((a, b) => (b.psi ?? 0) - (a.psi ?? 0));
  const binLabel = (i: number, cuts: number[]) =>
    i === 0 ? `< ${pct(cuts[0], 1)}` : i < cuts.length ? `${pct(cuts[i - 1], 1)}–${pct(cuts[i], 1)}` : i === cuts.length ? `≥ ${pct(cuts[i - 1], 1)}` : "missing";
  const scoreBins = run?.score
    ? run.score.ref_props
        .map((r, i) => ({ bin: binLabel(i, run.score!.cuts), reference: r, batch: run.score!.new_props[i] }))
        .filter((b) => b.bin !== "missing" || b.reference > 0 || b.batch > 0)
    : [];

  const upload = async (file: File) => {
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch("/api/proxy/v1/monitoring/check", { method: "POST", body: fd });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error?.message ?? `Check failed (${res.status})`);
      toast.success(`Drift check complete: ${body.result.status} (${body.result.alerts.length} alert(s))`);
      await qc.invalidateQueries();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const replayDemo = async () => {
    setBusy(true);
    try {
      // the API picks the demo batch that matches the served model (synthetic or real data)
      const res = await fetch("/api/proxy/v1/monitoring/replay-demo", { method: "POST" });
      const body = await res.json();
      if (!res.ok) throw new Error(body?.error?.message ?? `Replay failed (${res.status})`);
      toast.success(`Drift check complete: ${body.result.status} (${body.result.alerts.length} alert(s))`);
      await qc.invalidateQueries();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <PageHeader
        title="Monitoring"
        description="Input drift (PSI per feature), PD drift and missingness against the served model's training reference. Checks run on a schedule; you can also check a CSV batch here."
        actions={
          <>
            <input ref={fileRef} type="file" accept=".csv,text/csv" className="hidden" onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} aria-label="Upload a CSV batch" />
            <Button variant="secondary" onClick={replayDemo} disabled={busy} data-testid="replay-demo">
              Replay shifted demo batch
            </Button>
            <Button variant="primary" onClick={() => fileRef.current?.click()} disabled={busy}>
              <Upload /> {busy ? "Checking…" : "Check a CSV batch"}
            </Button>
            {meta.data?.grafana_url ? (
              <Button variant="ghost" asChild>
                <a href={meta.data.grafana_url} target="_blank" rel="noreferrer">
                  Grafana <ExternalLink />
                </a>
              </Button>
            ) : null}
          </>
        }
      />
      {drift.error ? <ErrorState error={drift.error} className="mb-4" /> : null}
      {!drift.isLoading && !run ? (
        <Card>
          <EmptyState title="No monitoring runs yet" body="Replay the shifted demo batch to see drift alerts fire, or drop CSVs into the monitor's inbox (MONITOR_INBOX) for the scheduled job." action={<Button variant="primary" onClick={replayDemo} disabled={busy}>Replay shifted demo batch</Button>} />
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5" data-testid="drift-summary">
            <div className="card flex flex-col gap-2 p-4">
              <span className="text-xs font-medium text-fg-secondary">Latest status</span>
              {run ? <StatusBadge status={run.status} /> : <div className="skeleton h-6 w-20" />}
              <span className="mt-auto truncate text-xs text-fg-muted">{run ? `${run.batch_name} · ${dateTime(run.created_at)}` : ""}</span>
            </div>
            <KpiCard label="Max feature PSI" value={dec(run?.max_feature_psi)} provenance={prov(p, "max_feature_psi")} sub={`${run?.n_significant ?? 0} significant (≥ 0.25)`} loading={drift.isLoading} />
            <KpiCard label="PD distribution PSI" value={dec(run?.score_psi)} provenance={prov(p, "score_psi")} sub={`mean PD ${pct(run?.ref_mean_pd)} → ${pct(run?.mean_pd)}`} loading={drift.isLoading} />
            <KpiCard label="Batch rows" value={num(run?.n_rows)} provenance={prov(p, "n_rows")} sub={run?.source} loading={drift.isLoading} />
            <KpiCard label="ROC-AUC on batch" value={run?.labels_available ? dec(run?.roc_auc) : "no labels"} provenance={prov(p, "roc_auc")} sub={run?.labels_available ? `observed DR ${pct(run?.observed_default_rate)}` : "performance needs labels"} loading={drift.isLoading} />
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
            <ChartCard
              title="PSI by input"
              description="Dashed lines: 0.10 moderate, 0.25 significant (rules of thumb)"
              provenance={prov(p, "psi")}
              loading={drift.isLoading}
              height={Math.max(240, features.length * 20 + 40)}
              table={{ columns: ["Input", "PSI", "Status"], rows: features.map((f) => [f.column, dec(f.psi), f.status]) }}
            >
              <ResponsiveContainer>
                <BarChart data={features} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 8 }}>
                  <CartesianGrid {...gridProps} horizontal={false} vertical />
                  <XAxis type="number" {...axisProps} domain={[0, (max: number) => Math.max(0.3, max * 1.1)]} tickFormatter={(v: number) => v.toFixed(2)} />
                  <YAxis type="category" dataKey="column" {...axisProps} width={170} interval={0} tick={{ fill: "var(--fg-secondary)", fontSize: 11 }} />
                  <ReferenceLine x={0.1} stroke="var(--warning)" strokeDasharray="4 4" />
                  <ReferenceLine x={0.25} stroke="var(--critical)" strokeDasharray="4 4" />
                  <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => dec(Number(v))} />} />
                  <Bar dataKey="psi" name="PSI" radius={[0, 4, 4, 0]} maxBarSize={14} isAnimationActive={false}>
                    {features.map((f) => (
                      <Cell key={f.column} fill={(f.psi ?? 0) >= 0.25 ? "var(--critical)" : (f.psi ?? 0) >= 0.1 ? "var(--warning)" : SERIES[0]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard
              title="PD distribution: reference vs batch"
              description="Share of applicants per validation-PD decile bin"
              provenance={prov(p, "new_props")}
              loading={drift.isLoading}
              legend={<Legend items={[{ label: "Reference (validation)", color: SERIES[0] }, { label: "Batch", color: SERIES[1] }]} />}
              table={{ columns: ["Bin", "Reference", "Batch"], rows: scoreBins.map((b) => [b.bin, pct(b.reference), pct(b.batch)]) }}
            >
              {scoreBins.length ? (
                <ResponsiveContainer>
                  <BarChart data={scoreBins} margin={{ top: 4, right: 8, bottom: 0, left: -8 }} barGap={2}>
                    <CartesianGrid {...gridProps} />
                    <XAxis dataKey="bin" {...axisProps} interval={0} tick={{ fill: "var(--chart-axis)", fontSize: 10 }} />
                    <YAxis {...axisProps} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} />
                    <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => pct(Number(v))} />} />
                    <Bar dataKey="reference" name="Reference" fill={SERIES[0]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    <Bar dataKey="batch" name="Batch" fill={SERIES[1]} radius={[4, 4, 0, 0]} isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState title="No PD comparison" body="This run had no score reference." />
              )}
            </ChartCard>
          </div>
          <Card className="mt-4">
            <CardHeader title="Inputs: drift and missingness" description="Latest run" />
            <DataTable columns={featCols} data={run?.features} loading={drift.isLoading} csvName="creditiq-drift-features" initialSorting={[{ id: "psi", desc: true }]} pageSize={15} />
          </Card>
        </>
      )}
      <Card className="mt-4">
        <CardHeader title="Alert history" description="Every alert from logged monitoring runs" />
        <DataTable columns={alertCols} data={alerts.data?.result as Alert[] | undefined} loading={alerts.isLoading} csvName="creditiq-alerts" emptyTitle="No alerts" emptyBody="Alerts appear when a check crosses a threshold." />
      </Card>
      <Card className="mt-4">
        <CardHeader title="Runs" description="Scheduled, audit-log, API and replay checks" />
        <DataTable
          columns={[
            { accessorKey: "created_at", header: "Time", cell: (c) => <span className="font-mono text-xs text-fg-secondary">{dateTime(c.getValue())}</span> },
            { accessorKey: "status", header: "Status", cell: (c) => <StatusBadge status={String(c.getValue())} /> },
            { accessorKey: "source", header: "Source" },
            { accessorKey: "batch_name", header: "Batch" },
            { accessorKey: "n_rows", header: "Rows", meta: { numeric: true }, cell: (c) => num(c.getValue()) },
            { accessorKey: "max_feature_psi", header: "Max PSI", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
            { accessorKey: "score_psi", header: "PD PSI", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
          ] as ColumnDef<Run, unknown>[]}
          data={history.data?.result as Run[] | undefined}
          loading={history.isLoading}
          csvName="creditiq-monitoring-runs"
        />
      </Card>
    </>
  );
}
