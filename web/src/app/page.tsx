"use client";

import Link from "next/link";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { Activity, ArrowRight, Gauge } from "lucide-react";
import { ChartCard, ChartTooltip, axisProps, gridProps } from "@/components/charts/chart";
import { BAND_VAR, KpiCard, PageHeader, StatusBadge, prov } from "@/components/ui/domain";
import { Button, Card, CardHeader, EmptyState, ErrorState } from "@/components/ui/primitives";
import { usePortfolio, useOverview } from "@/lib/api/hooks";
import { dateTime, money, pct } from "@/lib/format";

type Alert = { id: number; code: string; severity: string; message: string; created_at: string };
type Overview = {
  model_version: string;
  model_label: string;
  policy_version: string;
  portfolio_ead: number;
  expected_loss_12m: number;
  el_rate: number;
  observed_default_rate: number;
  avg_pd: number;
  flag_rate_test: number;
  test_roc_auc: number;
  test_pr_auc: number;
  n_loans: number;
  decisions_total: number;
  trend: { day: string; decisions: number; avg_pd: number; flag_rate: number | null }[];
  drift_status: string;
  drift_checked_at: string | null;
  recent_alerts: Alert[];
};

function delta(trend: Overview["trend"], key: "decisions" | "avg_pd" | "flag_rate") {
  if (trend.length < 2) return null;
  const a = trend[trend.length - 2][key];
  const b = trend[trend.length - 1][key];
  return a === null || b === null ? null : b - a;
}

export default function OverviewPage() {
  const { data, isLoading, error } = useOverview();
  const bands = usePortfolio("bands", { split: "test" });
  const o = data?.result as Overview | undefined;
  const p = data?.provenance;
  const trend = o?.trend ?? [];
  const dDecisions = delta(trend, "decisions");
  const dFlag = delta(trend, "flag_rate");

  return (
    <>
      <PageHeader
        title="Overview"
        description={
          o ? (
            <>
              Served model <span className="font-mono text-fg-secondary">v{o.model_version}</span> ({o.model_label}) under policy{" "}
              <span className="font-mono text-fg-secondary">{o.policy_version}</span>. Portfolio figures use the held-out test applicants.
            </>
          ) : (
            "Portfolio, model and monitoring at a glance."
          )
        }
        actions={
          <Button variant="primary" asChild>
            <Link href="/score">
              <Gauge /> Score an applicant
            </Link>
          </Button>
        }
      />
      {error ? <ErrorState error={error} className="mb-4" /> : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
        <KpiCard label="Portfolio EAD" loading={isLoading} value={money(o?.portfolio_ead)} provenance={prov(p, "portfolio_ead")} sub={`${(o?.n_loans ?? 0).toLocaleString()} loans`} />
        <KpiCard label="12-month expected loss" loading={isLoading} value={money(o?.expected_loss_12m)} provenance={prov(p, "expected_loss_12m")} sub="Σ PD × LGD × EAD" />
        <KpiCard label="EL rate" loading={isLoading} value={pct(o?.el_rate, 2)} provenance={prov(p, "el_rate")} sub={`observed default rate ${pct(o?.observed_default_rate)}`} />
        <KpiCard
          label="Flag rate"
          loading={isLoading}
          value={pct(o?.flag_rate_test)}
          provenance={prov(p, "flag_rate_test")}
          spark={trend.map((t) => t.flag_rate ?? 0)}
          delta={dFlag}
          deltaLabel={dFlag === null ? undefined : `${(dFlag * 100).toFixed(1)} pp vs previous day (live)`}
          goodWhen="neutral"
          sub="held-out applicants, active policy"
        />
        <KpiCard label="Model ROC-AUC (test)" loading={isLoading} value={o ? o.test_roc_auc.toFixed(3) : "—"} provenance={prov(p, "test_roc_auc")} sub={`PR-AUC ${o ? o.test_pr_auc.toFixed(3) : "—"}`} />
        <div className="card flex flex-col gap-2 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-fg-secondary">Drift status</span>
            <Activity aria-hidden className="size-4 text-fg-muted" />
          </div>
          {isLoading ? <div className="skeleton h-7 w-24" /> : <div className="mt-1">{o?.drift_status === "no checks yet" ? <span className="text-sm text-fg-muted">No checks yet</span> : <StatusBadge status={o?.drift_status ?? "ok"} />}</div>}
          <div className="mt-auto text-xs text-fg-muted">{o?.drift_checked_at ? `checked ${dateTime(o.drift_checked_at)}` : "run a monitoring check"}</div>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ChartCard
          title="Decisions per day"
          description={dDecisions === null ? "Live scoring requests (last 14 days)" : `Live scoring requests · ${dDecisions >= 0 ? "+" : ""}${dDecisions} vs previous day`}
          provenance={prov(p, "decisions")}
          loading={isLoading}
          table={{ columns: ["Day", "Decisions", "Flag rate"], rows: trend.map((t) => [t.day, t.decisions, pct(t.flag_rate)]) }}
        >
          {trend.length ? (
            <ResponsiveContainer>
              <BarChart data={trend} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="day" {...axisProps} tickFormatter={(d: string) => d.slice(5)} />
                <YAxis {...axisProps} allowDecimals={false} />
                <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip />} />
                <Bar dataKey="decisions" name="Decisions" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={28} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No decisions yet" body="Score an applicant or send a batch; each request is written to the audit log." />
          )}
        </ChartCard>
        <ChartCard
          title="Live flag rate"
          description="Share of scored applicants flagged for review, per day"
          provenance={prov(p, "flag_rate")}
          loading={isLoading}
          table={{ columns: ["Day", "Flag rate"], rows: trend.map((t) => [t.day, pct(t.flag_rate)]) }}
        >
          {trend.length ? (
            <ResponsiveContainer>
              <LineChart data={trend} margin={{ top: 4, right: 8, bottom: 0, left: -12 }}>
                <CartesianGrid {...gridProps} />
                <XAxis dataKey="day" {...axisProps} tickFormatter={(d: string) => d.slice(5)} />
                <YAxis {...axisProps} tickFormatter={(v: number) => `${Math.round(v * 100)}%`} domain={[0, "auto"]} />
                <RTooltip content={<ChartTooltip format={(v) => pct(Number(v))} />} />
                <Line type="monotone" dataKey="flag_rate" name="Flag rate" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No live decisions" body="The trend fills in as applicants are scored." />
          )}
        </ChartCard>
        <ChartCard
          title="Expected loss by risk band"
          description="Held-out test portfolio"
          provenance={prov(bands.data?.provenance as Record<string, string[]>, "expected_loss")}
          loading={bands.isLoading}
          table={{
            columns: ["Band", "Loans", "EAD", "Expected loss"],
            rows: (bands.data?.rows ?? []).map((r) => [String(r.risk_band), Number(r.loans), money(r.ead), money(r.expected_loss)]),
          }}
        >
          <ResponsiveContainer>
            <BarChart data={bands.data?.rows ?? []} margin={{ top: 4, right: 4, bottom: 0, left: -4 }}>
              <CartesianGrid {...gridProps} />
              <XAxis dataKey="risk_band" {...axisProps} />
              <YAxis {...axisProps} tickFormatter={(v: number) => money(v)} />
              <RTooltip cursor={{ fill: "var(--surface-2)" }} content={<ChartTooltip format={(v) => money(Number(v))} />} />
              <Bar dataKey="expected_loss" name="Expected loss" radius={[4, 4, 0, 0]} maxBarSize={44}>
                {(bands.data?.rows ?? []).map((r) => (
                  <Cell key={String(r.risk_band)} fill={BAND_VAR[String(r.risk_band)] ?? "var(--series-1)"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <Card className="mt-4">
        <CardHeader
          title="Recent alerts"
          description="From the latest monitoring run"
          actions={
            <Button variant="ghost" size="sm" asChild>
              <Link href="/monitoring">
                Monitoring <ArrowRight />
              </Link>
            </Button>
          }
        />
        {o?.recent_alerts?.length ? (
          <ul className="divide-y divide-border">
            {o.recent_alerts.map((a) => (
              <li key={a.id} className="flex flex-wrap items-center gap-3 px-4 py-2.5 text-[13px]">
                <StatusBadge status={a.severity} />
                <span className="text-fg">{a.message}</span>
                <span className="ml-auto font-mono text-xs text-fg-muted">{a.code}</span>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState title="No alerts" body="Drift checks run on a schedule; replay the shifted demo batch from the Monitoring page to see alerts fire." />
        )}
      </Card>
    </>
  );
}
