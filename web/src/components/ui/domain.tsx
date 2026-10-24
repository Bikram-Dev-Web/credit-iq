"use client";

import * as React from "react";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { AlertTriangle, CheckCircle2, CircleAlert, OctagonAlert, X } from "lucide-react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";
import { cn } from "@/lib/utils";
import { Tooltip } from "./primitives";

/* ------------------------------------------------------------------ Provenance */
export type ProvenanceKind = "observed" | "measured" | "model" | "assumption" | "simulated" | "synthetic";

const PROV: Record<ProvenanceKind, { label: string; help: string }> = {
  observed: { label: "Observed", help: "A historical fact from the data (labels, counts, exposures)." },
  measured: { label: "Measured", help: "Computed by evaluating the model on held-out data it did not train on." },
  model: { label: "Model", help: "Produced by the trained model (PD, raw score, explanation)." },
  assumption: { label: "Assumption", help: "A value chosen by a person (LGD, CCF, rates, thresholds, costs)." },
  simulated: { label: "Simulated", help: "A projection combining model outputs with assumptions." },
  synthetic: { label: "Synthetic", help: "Computed on fabricated demonstration data, not real borrowers." },
};

export function ProvenancePill({ kind, className }: { kind: ProvenanceKind; className?: string }) {
  const p = PROV[kind] ?? { label: kind, help: "" };
  return (
    <Tooltip content={<span><strong>{p.label}.</strong> {p.help}</span>}>
      <span
        tabIndex={0}
        style={{ color: `var(--prov-${kind})`, borderColor: `color-mix(in srgb, var(--prov-${kind}) 35%, transparent)` }}
        className={cn(
          "inline-flex h-[18px] cursor-help items-center rounded-full border px-1.5 text-[10px] font-semibold uppercase leading-none tracking-wide",
          className,
        )}
      >
        {p.label}
      </span>
    </Tooltip>
  );
}

export function Provenance({ labels, className }: { labels?: readonly string[] | null; className?: string }) {
  if (!labels?.length) return null;
  return (
    <span className={cn("inline-flex flex-wrap items-center gap-1", className)}>
      {labels.map((l) => (
        <ProvenancePill key={l} kind={l as ProvenanceKind} />
      ))}
    </span>
  );
}

/** Look up a field's labels in a response's provenance map. */
export function prov(map: Record<string, readonly string[]> | undefined | null, key: string): readonly string[] {
  return map?.[key] ?? [];
}

/* ------------------------------------------------------------------ Bands, outcomes, status */
export const BAND_VAR: Record<string, string> = {
  Low: "var(--band-low)",
  Moderate: "var(--band-moderate)",
  Elevated: "var(--band-elevated)",
  High: "var(--band-high)",
};

export function BandBadge({ band }: { band: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-surface-2 px-2 py-0.5 text-[11px] font-medium text-fg">
      <span aria-hidden className="size-2 rounded-full" style={{ background: BAND_VAR[band] ?? "var(--fg-muted)" }} />
      {band}
    </span>
  );
}

export function OutcomeBadge({ outcome }: { outcome: string }) {
  const flagged = outcome === "flagged_for_review";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium text-fg",
        flagged ? "bg-serious-soft" : "bg-good-soft",
      )}
    >
      {flagged ? (
        <AlertTriangle aria-hidden className="size-3" style={{ color: "var(--serious)" }} />
      ) : (
        <CheckCircle2 aria-hidden className="size-3" style={{ color: "var(--good)" }} />
      )}
      {flagged ? "Flagged for review" : "Within threshold"}
    </span>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { tone: string; Icon: React.ElementType; label: string }> = {
    ok: { tone: "good", Icon: CheckCircle2, label: "Stable" },
    warning: { tone: "warning", Icon: CircleAlert, label: "Warning" },
    critical: { tone: "critical", Icon: OctagonAlert, label: "Critical" },
  };
  const s = map[status] ?? { tone: "neutral", Icon: CircleAlert, label: status };
  return (
    <span
      className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium text-fg"
      style={{ background: s.tone === "neutral" ? "var(--surface-2)" : `var(--${s.tone}-soft)` }}
    >
      <s.Icon aria-hidden className="size-3" style={{ color: s.tone === "neutral" ? "var(--fg-muted)" : `var(--${s.tone})` }} />
      {s.label}
    </span>
  );
}

/* ------------------------------------------------------------------ KPI card */
export function KpiCard({
  label,
  value,
  provenance,
  delta,
  deltaLabel,
  goodWhen = "up",
  spark,
  sub,
  loading,
}: {
  label: string;
  value: React.ReactNode;
  provenance?: readonly string[];
  delta?: number | null;
  deltaLabel?: string;
  goodWhen?: "up" | "down" | "neutral";
  spark?: number[];
  sub?: React.ReactNode;
  loading?: boolean;
}) {
  const id = React.useId().replace(/:/g, "");
  const tone =
    delta === null || delta === undefined || goodWhen === "neutral" || delta === 0
      ? "var(--fg-muted)"
      : (delta > 0) === (goodWhen === "up")
        ? "var(--good)"
        : "var(--critical)";
  return (
    <div className="card flex min-w-0 flex-col gap-2 p-4">
      <div className="flex flex-col gap-1.5">
        <span className="text-xs font-medium text-fg-secondary">{label}</span>
        <Provenance labels={provenance} className="min-h-[18px]" />
      </div>
      {loading ? (
        <div className="skeleton h-7 w-28" />
      ) : (
        <div className="num font-mono text-2xl font-semibold tracking-tight text-fg">{value}</div>
      )}
      <div className="flex min-h-8 items-end justify-between gap-3">
        <div className="text-xs text-fg-muted">
          {delta !== undefined && delta !== null ? (
            <span className="num font-mono" style={{ color: tone }}>
              {delta > 0 ? "▲" : delta < 0 ? "▼" : "■"} {deltaLabel}
            </span>
          ) : (
            sub
          )}
        </div>
        {spark && spark.length > 1 ? (
          <div className="h-8 w-24 shrink-0" aria-hidden>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={spark.map((v, i) => ({ i, v }))} margin={{ top: 2, bottom: 2, left: 0, right: 0 }}>
                <defs>
                  <linearGradient id={`sp-${id}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="v" stroke="var(--accent)" strokeWidth={1.5} fill={`url(#sp-${id})`} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ Dialog and side sheet */
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  side = false,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: string;
  description?: string;
  children: React.ReactNode;
  side?: boolean;
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" />
        <DialogPrimitive.Content
          className={cn(
            "fixed z-50 flex flex-col border border-border bg-surface shadow-card focus:outline-none",
            side
              ? "inset-y-0 right-0 w-full max-w-xl overflow-y-auto border-y-0 border-r-0"
              : "left-1/2 top-1/2 max-h-[85vh] w-[calc(100vw-2rem)] max-w-lg -translate-x-1/2 -translate-y-1/2 overflow-y-auto rounded-lg",
          )}
        >
          <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
            <div>
              <DialogPrimitive.Title className="text-sm font-semibold text-fg">{title}</DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="mt-0.5 text-xs text-fg-muted">{description}</DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">{title}</DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close className="rounded p-1 text-fg-muted hover:bg-surface-2 hover:text-fg" aria-label="Close">
              <X className="size-4" />
            </DialogPrimitive.Close>
          </div>
          <div className="p-5">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

/* ------------------------------------------------------------------ Page header */
export function PageHeader({
  title,
  description,
  actions,
  badges,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  badges?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
          {badges}
        </div>
        {description ? <p className="mt-1 max-w-3xl text-[13px] text-fg-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function SectionGrid({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("grid gap-4", className)}>{children}</div>;
}
