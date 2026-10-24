"use client";

import * as React from "react";
import { ShieldCheck } from "lucide-react";
import { DataTable, type ColumnDef } from "@/components/ui/data-table";
import { BandBadge, Dialog, OutcomeBadge, PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Card, ErrorState, Input, Select, Stat } from "@/components/ui/primitives";
import { useDecisions } from "@/lib/api/hooks";
import { dateTime, money, pct, shortId } from "@/lib/format";

type Decision = {
  request_id: string;
  created_at: string;
  endpoint: string;
  batch_id: string | null;
  api_key_id: string;
  role: string;
  model_name: string;
  model_version: string;
  policy_version: string;
  pd: number;
  raw_score: number;
  band: string;
  outcome: string;
  threshold: number;
  threshold_source: string;
  rules_fired: { id: string; reason?: string; value?: number | string | null; limit?: unknown }[];
  top_reasons: { feature: string; contribution: number }[];
  ead: number;
  expected_loss: number;
  input_hash: string;
  hash_key_id: string;
  latency_ms: number | null;
};

const columns: ColumnDef<Decision, unknown>[] = [
  { accessorKey: "created_at", header: "Time", cell: (c) => <span className="font-mono text-xs text-fg-secondary">{dateTime(c.getValue())}</span> },
  { accessorKey: "request_id", header: "Request", cell: (c) => <span className="font-mono text-xs">{shortId(c.getValue(), 10)}</span> },
  { accessorKey: "endpoint", header: "Endpoint" },
  { accessorKey: "outcome", header: "Outcome", cell: (c) => <OutcomeBadge outcome={String(c.getValue())} /> },
  { accessorKey: "band", header: "Band", cell: (c) => <BandBadge band={String(c.getValue())} /> },
  { accessorKey: "pd", header: "PD", meta: { numeric: true }, cell: (c) => pct(c.getValue(), 2) },
  { accessorKey: "threshold", header: "Threshold", meta: { numeric: true }, cell: (c) => pct(c.getValue(), 1) },
  { accessorKey: "expected_loss", header: "Exp. loss", meta: { numeric: true }, cell: (c) => money(c.getValue()) },
  { accessorKey: "model_version", header: "Model", cell: (c) => <span className="font-mono text-xs">v{String(c.getValue())}</span> },
  { accessorKey: "policy_version", header: "Policy", cell: (c) => <span className="font-mono text-xs">{String(c.getValue())}</span> },
  {
    id: "rules",
    header: "Rules fired",
    accessorFn: (d) => d.rules_fired.map((r) => r.id).join(", "),
    cell: (c) => <span className="font-mono text-xs text-fg-secondary">{String(c.getValue()) || "—"}</span>,
  },
];

export default function DecisionsPage() {
  const [q, setQ] = React.useState("");
  const [debounced, setDebounced] = React.useState("");
  const [outcome, setOutcome] = React.useState<"all" | "within_threshold" | "flagged_for_review">("all");
  const [selected, setSelected] = React.useState<Decision | null>(null);
  React.useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 300);
    return () => clearTimeout(t);
  }, [q]);
  const { data, isLoading, error } = useDecisions({ q: debounced || undefined, outcome: outcome === "all" ? undefined : outcome, limit: 500 });
  const res = data?.result as { total: number; items: Decision[] } | undefined;
  const p = data?.provenance as Record<string, string[]> | undefined;

  return (
    <>
      <PageHeader
        title="Decisions log"
        description="Every scoring request, newest first. The log holds scores, outcomes, rules fired and an HMAC of the inputs — never the applicant's inputs themselves."
        badges={
          <span className="inline-flex items-center gap-1 text-xs text-fg-muted">
            <ShieldCheck className="size-3.5 text-accent" /> privacy-preserving audit
          </span>
        }
      />
      {error ? <ErrorState error={error} className="mb-4" /> : null}
      <Card>
        <DataTable
          columns={columns}
          data={res?.items}
          loading={isLoading}
          csvName="creditiq-decisions"
          onRowClick={setSelected}
          rowLabel={(d) => `Open decision ${d.request_id}`}
          searchPlaceholder="Filter loaded rows…"
          emptyTitle="No decisions match"
          emptyBody="Score an applicant on the Score page, or send a batch to /v1/score/batch."
          toolbar={
            <>
              <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search request id, batch id or input hash" aria-label="Search the audit log" className="h-7 w-72 max-w-full text-xs" />
              <div className="w-44">
                <Select
                  value={outcome}
                  ariaLabel="Outcome filter"
                  onValueChange={(v) => setOutcome(v as typeof outcome)}
                  options={[
                    { value: "all", label: "All outcomes" },
                    { value: "within_threshold", label: "Within threshold" },
                    { value: "flagged_for_review", label: "Flagged for review" },
                  ]}
                  className="h-7 text-xs"
                />
              </div>
              <span className="num text-xs text-fg-muted">{res ? `${res.total.toLocaleString()} total` : ""}</span>
            </>
          }
        />
      </Card>
      <Dialog open={selected !== null} onOpenChange={(o) => !o && setSelected(null)} title="Decision" description={selected?.request_id} side>
        {selected ? (
          <div className="flex flex-col gap-5 text-[13px]">
            <div className="flex flex-wrap items-center gap-2">
              <OutcomeBadge outcome={selected.outcome} />
              <BandBadge band={selected.band} />
              <span className="text-xs text-fg-muted">{dateTime(selected.created_at)} · {selected.endpoint}</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <Stat label="PD" provenance={<Provenance labels={prov(p, "pd")} />} value={pct(selected.pd, 2)} />
              <Stat label="Threshold" provenance={<Provenance labels={prov(p, "threshold")} />} value={pct(selected.threshold)} sub={selected.threshold_source} />
              <Stat label="Expected loss" provenance={<Provenance labels={prov(p, "expected_loss")} />} value={money(selected.expected_loss)} />
              <Stat label="EAD" provenance={<Provenance labels={prov(p, "ead")} />} value={money(selected.ead)} />
            </div>
            <dl className="grid grid-cols-[8rem_1fr] gap-x-4 gap-y-1.5 text-xs">
              <dt className="text-fg-muted">Model</dt>
              <dd className="font-mono">{selected.model_name} v{selected.model_version}</dd>
              <dt className="text-fg-muted">Policy</dt>
              <dd className="font-mono">{selected.policy_version}</dd>
              <dt className="text-fg-muted">API key</dt>
              <dd className="font-mono">{selected.api_key_id} ({selected.role})</dd>
              <dt className="text-fg-muted">Input HMAC</dt>
              <dd className="break-all font-mono">{selected.input_hash}</dd>
              <dt className="text-fg-muted">Latency</dt>
              <dd className="font-mono">{selected.latency_ms === null ? "—" : `${selected.latency_ms.toFixed(1)} ms`}</dd>
            </dl>
            <div>
              <h4 className="mb-2 text-xs font-semibold text-fg">Rules fired</h4>
              {selected.rules_fired.length ? (
                <ul className="flex flex-col gap-1.5">
                  {selected.rules_fired.map((r) => (
                    <li key={r.id} className="rounded-md border border-border bg-surface-2 px-3 py-2 text-xs">
                      <span className="font-mono text-fg">{r.id}</span>
                      {r.reason ? <span className="text-fg-muted"> — {r.reason}</span> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-fg-muted">None.</p>
              )}
            </div>
            <div>
              <h4 className="mb-2 text-xs font-semibold text-fg">Top reasons (signed contributions)</h4>
              {selected.top_reasons.length ? (
                <ul className="flex flex-col gap-1">
                  {selected.top_reasons.map((r) => (
                    <li key={r.feature} className="flex items-center gap-2 text-xs">
                      <span className="w-48 truncate text-fg-secondary">{r.feature}</span>
                      <span className="relative h-2 flex-1 rounded bg-surface-2">
                        <span
                          className="absolute top-0 h-2 rounded"
                          style={{
                            left: r.contribution >= 0 ? "50%" : `${50 - Math.min(50, Math.abs(r.contribution) * 25)}%`,
                            width: `${Math.min(50, Math.abs(r.contribution) * 25)}%`,
                            background: r.contribution >= 0 ? "var(--diverging-neg)" : "var(--diverging-pos)",
                          }}
                        />
                      </span>
                      <span className="num w-16 text-right font-mono">{r.contribution > 0 ? "+" : ""}{r.contribution.toFixed(3)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-fg-muted">Not computed for batch rows.</p>
              )}
            </div>
            <p className="rounded-md border border-border bg-surface-2 px-3 py-2 text-[11px] text-fg-muted">
              The applicant&apos;s inputs are not stored. To check whether a known application was scored, recompute its HMAC with the deployment&apos;s audit key and search for it.
            </p>
          </div>
        ) : null}
      </Dialog>
    </>
  );
}
