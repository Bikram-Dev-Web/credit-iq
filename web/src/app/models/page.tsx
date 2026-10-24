"use client";

import * as React from "react";
import { CartesianGrid, Line, LineChart, ReferenceDot, ResponsiveContainer, Scatter, ScatterChart, Tooltip as RTooltip, XAxis, YAxis, ZAxis } from "recharts";
import { ArrowUpCircle, Rocket } from "lucide-react";
import { toast } from "sonner";
import { ChartCard, ChartTooltip, Legend, SERIES, axisProps, gridProps } from "@/components/charts/chart";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { Dialog, KpiCard, PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Badge, Button, Card, CardBody, CardHeader, ErrorState, Slider, Stat } from "@/components/ui/primitives";
import { type ModelList, useEvaluation, useModel, useModelCard, useModels, usePromote, useSession } from "@/lib/api/hooks";
import { dateTime, dec, num, pct } from "@/lib/format";

type Version = ModelList["versions"][number];
type Pt = Record<string, number>;
type SplitEval = { metrics: Record<string, number | null>; roc: Pt[]; pr: Pt[]; calibration: Pt[]; ece: number | null };
type ModelEval = { label: string; threshold: number; is_calibrated: boolean; splits: { validation: SplitEval; test: SplitEval }; threshold_sweep_validation: Pt[] };
type Evaluation = { selected: string; models: Record<string, ModelEval>; comparison: Record<string, unknown>[]; split_sizes: Record<string, number> };

function Markdown({ text }: { text: string }) {
  // Minimal renderer for the generated model card: headings, bullets, tables, emphasis.
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let table: string[][] = [];
  const flush = () => {
    if (table.length) {
      const [head, , ...body] = table;
      out.push(
        <div key={out.length} className="my-2 overflow-x-auto">
          <table className="text-xs">
            <thead><tr>{head.map((h, i) => <th key={i} className="border-b border-border px-2 py-1 text-left font-medium text-fg-muted">{h}</th>)}</tr></thead>
            <tbody>{body.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="num border-b border-border/50 px-2 py-1 font-mono">{c}</td>)}</tr>)}</tbody>
          </table>
        </div>,
      );
      table = [];
    }
  };
  const inline = (s: string) => s.replace(/\*\*(.+?)\*\*/g, "$1").replace(/`(.+?)`/g, "$1").replace(/_(.+?)_/g, "$1");
  for (const l of lines) {
    if (l.startsWith("|")) {
      table.push(l.split("|").slice(1, -1).map((c) => c.trim()));
      continue;
    }
    flush();
    if (l.startsWith("# ")) out.push(<h2 key={out.length} className="mb-2 text-base font-semibold text-fg">{inline(l.slice(2))}</h2>);
    else if (l.startsWith("## ") || l.startsWith("### ")) out.push(<h3 key={out.length} className="mb-1 mt-4 text-[13px] font-semibold text-fg">{inline(l.replace(/^#+ /, ""))}</h3>);
    else if (l.startsWith("- ")) out.push(<li key={out.length} className="ml-4 list-disc text-xs text-fg-secondary">{inline(l.slice(2))}</li>);
    else if (l.trim()) out.push(<p key={out.length} className="text-xs text-fg-secondary">{inline(l)}</p>);
  }
  flush();
  return <div className="flex flex-col gap-1">{out}</div>;
}

export default function ModelsPage() {
  const models = useModels();
  const session = useSession();
  const promote = usePromote();
  const [version, setVersion] = React.useState<string | undefined>(undefined);
  const [confirm, setConfirm] = React.useState<Version | null>(null);
  const serving = models.data?.serving_version ?? undefined;
  const v = version ?? serving;
  const detail = useModel(v);
  const evaluation = useEvaluation(v);
  const card = useModelCard(v);
  const ev = evaluation.data?.result as Evaluation | undefined;
  const sel = ev ? ev.models[ev.selected] : undefined;
  const sweep = sel?.threshold_sweep_validation ?? [];
  // The explorer starts at the version's own threshold; a value the user picks applies to that version only.
  const [picked, setPicked] = React.useState<{ version?: string; t: number } | null>(null);
  const thr = picked && picked.version === v ? picked.t : sel ? Number(sel.threshold.toFixed(2)) : null;
  const setThr = (t: number) => setPicked({ version: v, t });
  const point = sweep.length && thr !== null ? sweep.reduce((a, b) => (Math.abs(b.threshold - thr) < Math.abs(a.threshold - thr) ? b : a)) : null;
  const mp = detail.data?.provenance as Record<string, string[]> | undefined;
  const d = detail.data?.result as Record<string, unknown> | undefined;
  const tm = (d?.test_metrics ?? {}) as Record<string, number>;
  const keys = ev ? Object.keys(ev.models) : [];

  const columns: ColumnDef<Version, unknown>[] = [
    { accessorKey: "version", header: "Version", cell: (c) => <span className="font-mono">v{String(c.getValue())}</span> },
    {
      accessorKey: "aliases",
      header: "Aliases",
      cell: (c) => (
        <span className="flex gap-1">
          {(c.getValue() as string[]).map((a) => (
            <Badge key={a} tone={a === "production" ? "accent" : "neutral"}>{a}</Badge>
          ))}
        </span>
      ),
    },
    { accessorKey: "selected_model", header: "Algorithm", cell: (c) => String(c.getValue() ?? "—") },
    { accessorKey: "test_roc_auc", header: "Test ROC-AUC", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
    { accessorKey: "test_pr_auc", header: "Test PR-AUC", meta: { numeric: true }, cell: (c) => dec(c.getValue()) },
    { accessorKey: "test_brier", header: "Brier", meta: { numeric: true }, cell: (c) => dec(c.getValue(), 4) },
    { accessorKey: "is_synthetic", header: "Data", cell: (c) => (c.getValue() ? <Badge tone="serious">synthetic</Badge> : <Badge>observed</Badge>) },
    { accessorKey: "created", header: "Registered", cell: (c) => <span className="text-xs text-fg-secondary">{c.getValue() ? dateTime(new Date(Number(c.getValue())).toISOString()) : "—"}</span> },
    {
      id: "actions",
      header: "",
      enableHiding: false,
      cell: (c) =>
        c.row.original.serving ? (
          <Badge tone="good">serving</Badge>
        ) : session.data?.role === "admin" ? (
          <Button size="sm" variant="secondary" data-testid={`promote-${c.row.original.version}`} onClick={(e) => { e.stopPropagation(); setConfirm(c.row.original); }}>
            <ArrowUpCircle /> Promote
          </Button>
        ) : null,
    },
  ];

  return (
    <>
      <PageHeader title="Models" description="Registered versions in the model registry. Click a version for its evaluation; admins can promote a version to production (hot-reloaded without dropping requests)." />
      {models.error ? <ErrorState error={models.error} className="mb-4" /> : null}
      <Card className="mb-4">
        <CardHeader title={`Registry · ${models.data?.model_name ?? ""}`} description={`Serving alias: ${models.data?.serving_alias ?? "production"}`} />
        <DataTable columns={columns} data={models.data?.versions} loading={models.isLoading} onRowClick={(r) => setVersion(r.version)} rowLabel={(r) => `Show model v${r.version}`} csvName="creditiq-models" emptyTitle="No registered versions" emptyBody="Train one: uv run creditiq-train train --config services/trainer/configs/synthetic.yaml" />
      </Card>

      {v ? (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-semibold text-fg">Version v{v}</h2>
            {sel ? <Badge>{sel.label}</Badge> : null}
            {d?.threshold_source ? <Badge tone="accent">threshold from {String(d.threshold_source)}</Badge> : null}
            {sel && !sel.is_calibrated ? <Badge tone="warning">uncalibrated</Badge> : null}
          </div>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <KpiCard label="Test ROC-AUC" value={dec(tm.roc_auc)} provenance={prov(mp, "roc_auc")} loading={detail.isLoading} sub={`Gini ${dec(tm.gini)}`} />
            <KpiCard label="Test PR-AUC" value={dec(tm.pr_auc)} provenance={prov(mp, "pr_auc")} loading={detail.isLoading} sub={`random baseline ${pct(tm.prevalence)}`} />
            <KpiCard label="Brier score" value={dec(tm.brier, 4)} provenance={prov(mp, "brier")} loading={detail.isLoading} sub="lower is better" />
            <KpiCard label="KS" value={dec(tm.ks)} provenance={prov(mp, "ks")} loading={detail.isLoading} />
            <KpiCard label="Threshold" value={pct(Number(d?.threshold))} provenance={prov(mp, "threshold")} loading={detail.isLoading} sub={String(d?.threshold_rule ?? "")} />
          </div>
          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <ChartCard
              title="ROC (test)"
              provenance={["measured"]}
              loading={evaluation.isLoading}
              legend={<Legend items={keys.map((k, i) => ({ label: ev!.models[k].label, color: SERIES[i % 3] }))} />}
              table={{ columns: ["Model", "Test ROC-AUC"], rows: keys.map((k) => [ev!.models[k].label, dec(ev!.models[k].splits.test.metrics.roc_auc)]) }}
            >
              <ResponsiveContainer>
                <LineChart margin={{ top: 4, right: 8, bottom: 8, left: -8 }}>
                  <CartesianGrid {...gridProps} vertical />
                  <XAxis type="number" dataKey="fpr" domain={[0, 1]} {...axisProps} label={{ value: "False positive rate", position: "insideBottom", offset: -4 }} />
                  <YAxis type="number" dataKey="tpr" domain={[0, 1]} {...axisProps} />
                  <RTooltip content={<ChartTooltip format={(v) => dec(Number(v))} labelFormat={(l) => `FPR ${dec(Number(l))}`} />} />
                  <Line data={[{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }]} dataKey="tpr" stroke="var(--border-strong)" strokeDasharray="4 4" dot={false} isAnimationActive={false} name="Random" />
                  {keys.map((k, i) => (
                    <Line key={k} data={ev!.models[k].splits.test.roc} dataKey="tpr" name={ev!.models[k].label} stroke={SERIES[i % 3]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard
              title="Precision–recall (test)"
              provenance={["measured"]}
              loading={evaluation.isLoading}
              legend={<Legend items={keys.map((k, i) => ({ label: ev!.models[k].label, color: SERIES[i % 3] }))} />}
              table={{ columns: ["Model", "Test PR-AUC"], rows: keys.map((k) => [ev!.models[k].label, dec(ev!.models[k].splits.test.metrics.pr_auc)]) }}
            >
              <ResponsiveContainer>
                <LineChart margin={{ top: 4, right: 8, bottom: 8, left: -8 }}>
                  <CartesianGrid {...gridProps} vertical />
                  <XAxis type="number" dataKey="recall" domain={[0, 1]} {...axisProps} label={{ value: "Recall", position: "insideBottom", offset: -4 }} />
                  <YAxis type="number" dataKey="precision" domain={[0, 1]} {...axisProps} />
                  <RTooltip content={<ChartTooltip format={(v) => dec(Number(v))} labelFormat={(l) => `Recall ${dec(Number(l))}`} />} />
                  {keys.map((k, i) => (
                    <Line key={k} data={ev!.models[k].splits.test.pr} dataKey="precision" name={ev!.models[k].label} stroke={SERIES[i % 3]} strokeWidth={2} dot={false} isAnimationActive={false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
            <ChartCard
              title="Calibration (test)"
              description={sel ? `Selected model · expected calibration error ${pct(sel.splits.test.ece, 2)}` : undefined}
              provenance={["measured"]}
              loading={evaluation.isLoading}
              legend={<Legend items={[{ label: "Observed default rate per bin", color: SERIES[0] }, { label: "Perfect calibration", color: "var(--border-strong)", dashed: true }]} />}
              table={{ columns: ["Mean PD", "Observed", "95% CI", "n"], rows: (sel?.splits.test.calibration ?? []).map((b) => [pct(b.mean_pd), pct(b.observed_rate), `${pct(b.ci_low)}–${pct(b.ci_high)}`, num(b.n)]) }}
            >
              <ResponsiveContainer>
                <ScatterChart margin={{ top: 4, right: 8, bottom: 8, left: -8 }}>
                  <CartesianGrid {...gridProps} vertical />
                  <XAxis type="number" dataKey="mean_pd" domain={[0, "dataMax"]} {...axisProps} tickFormatter={(x: number) => `${Math.round(x * 100)}%`} name="Mean PD" />
                  <YAxis type="number" dataKey="observed_rate" domain={[0, "dataMax"]} {...axisProps} tickFormatter={(x: number) => `${Math.round(x * 100)}%`} name="Observed" />
                  <ZAxis range={[60, 60]} />
                  <RTooltip content={({ active, payload }) => (active && payload?.length ? <ChartTooltip active label={`Mean PD ${pct(Number(payload[0].payload.mean_pd))}`} payload={[{ name: "Observed", value: Number(payload[0].payload.observed_rate), color: SERIES[0] }, { name: "n", value: Number(payload[0].payload.n), color: "transparent" }]} format={(x, k) => (k === "n" ? num(Number(x)) : pct(Number(x)))} /> : null)} />
                  <Scatter data={[{ mean_pd: 0, observed_rate: 0 }, { mean_pd: Math.max(...(sel?.splits.test.calibration ?? [{ mean_pd: 0.3 }]).map((b) => b.mean_pd)), observed_rate: Math.max(...(sel?.splits.test.calibration ?? [{ mean_pd: 0.3 }]).map((b) => b.mean_pd)) }]} line={{ stroke: "var(--border-strong)", strokeDasharray: "4 4" }} shape={() => <g />} isAnimationActive={false} />
                  <Scatter data={sel?.splits.test.calibration ?? []} fill={SERIES[0]} line={{ stroke: SERIES[0], strokeWidth: 2 }} isAnimationActive={false} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartCard>
            <Card>
              <CardHeader title="Threshold explorer (validation)" description="Thresholds are chosen on validation; the test split is reported, never tuned on. This view does not change the served threshold." actions={<Provenance labels={["measured"]} />} />
              <CardBody className="flex flex-col gap-4">
                {point ? (
                  <>
                    <div className="flex items-center gap-3">
                      <Slider aria-label="Threshold" min={sweep[0].threshold} max={sweep[sweep.length - 1].threshold} step={0.01} value={[thr ?? 0]} onValueChange={([x]) => setThr(x)} />
                      <span className="num w-16 text-right font-mono text-sm text-fg">{pct(point.threshold, 0)}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                      <Stat label="Flag rate" value={pct(point.flag_rate)} />
                      <Stat label="Recall" value={pct(point.recall)} />
                      <Stat label="Precision" value={pct(point.precision)} />
                      <Stat label="Specificity" value={pct(point.specificity)} />
                    </div>
                    <div className="grid grid-cols-4 gap-1 text-center text-[11px]">
                      {(["tp", "fp", "fn", "tn"] as const).map((k) => (
                        <div key={k} className="rounded-md border border-border bg-surface-2 px-2 py-1.5">
                          <div className="font-semibold uppercase text-fg-muted">{k}</div>
                          <div className="num font-mono text-fg">{num(point[k])}</div>
                        </div>
                      ))}
                    </div>
                    <div style={{ height: 140 }}>
                      <ResponsiveContainer>
                        <LineChart data={sweep} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                          <CartesianGrid {...gridProps} />
                          <XAxis dataKey="threshold" type="number" domain={[0, 1]} {...axisProps} tickFormatter={(x: number) => `${Math.round(x * 100)}%`} />
                          <YAxis {...axisProps} domain={[0, 1]} tickFormatter={(x: number) => `${Math.round(x * 100)}%`} />
                          <RTooltip content={<ChartTooltip format={(x) => pct(Number(x))} labelFormat={(l) => `Threshold ${pct(Number(l), 0)}`} />} />
                          <Line dataKey="recall" name="Recall" stroke={SERIES[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
                          <Line dataKey="precision" name="Precision" stroke={SERIES[1]} strokeWidth={2} dot={false} isAnimationActive={false} />
                          <ReferenceDot x={point.threshold} y={point.recall} r={4} fill={SERIES[0]} stroke="var(--surface)" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <Legend items={[{ label: "Recall", color: SERIES[0] }, { label: "Precision", color: SERIES[1] }]} />
                  </>
                ) : (
                  <div className="skeleton h-48" />
                )}
              </CardBody>
            </Card>
          </div>
          <Card className="mt-4">
            <CardHeader title="Model card" description="Generated from this version's training metadata" />
            <CardBody className="max-h-[520px] overflow-y-auto">{card.data ? <Markdown text={(card.data.result as { markdown: string }).markdown} /> : <div className="skeleton h-40" />}</CardBody>
          </Card>
        </>
      ) : null}

      <Dialog open={confirm !== null} onOpenChange={(o) => !o && setConfirm(null)} title={`Promote v${confirm?.version} to production?`} description="The current production version becomes 'previous' (one-step rollback). The API loads and warms the new version, then swaps it in.">
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={() => setConfirm(null)}>Cancel</Button>
          <Button
            variant="primary"
            data-testid="confirm-promote"
            disabled={promote.isPending}
            onClick={async () => {
              if (!confirm) return;
              try {
                const r = await promote.mutateAsync(confirm.version);
                toast.success(`v${r.version} is now serving (replaced v${r.replaced ?? "—"})`);
                setConfirm(null);
              } catch (e) {
                toast.error((e as Error).message);
              }
            }}
          >
            <Rocket /> {promote.isPending ? "Promoting…" : "Promote"}
          </Button>
        </div>
      </Dialog>
    </>
  );
}
