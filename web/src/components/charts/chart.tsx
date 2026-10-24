"use client";

import * as React from "react";
import { BarChart3, Table2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Provenance } from "@/components/ui/domain";
import { Button } from "@/components/ui/primitives";

/** Chart rules (docs/v2/DESIGN.md): one y-axis per chart, 2px lines, 4px bar ends, recessive grid, tooltips on
 * hover, a legend for 2+ series, and a table view of the same data. Colours come from tokens only. */

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"] as const;

export const axisProps = {
  tickLine: false,
  axisLine: false,
  tick: { fill: "var(--chart-axis)", fontSize: 11 },
  tickMargin: 6,
} as const;

export const gridProps = { stroke: "var(--chart-grid)", strokeDasharray: "0", vertical: false } as const;

type TooltipEntry = { name?: string | number; value?: number | string; color?: string; dataKey?: string | number; payload?: Record<string, unknown> };

export function ChartTooltip({
  active,
  payload,
  label,
  format = (v) => String(v),
  labelFormat,
}: {
  active?: boolean;
  payload?: TooltipEntry[];
  label?: string | number;
  format?: (v: number | string, key?: string) => string;
  labelFormat?: (l: string | number) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-border bg-surface-2 px-2.5 py-2 text-xs shadow-card">
      {label !== undefined ? (
        <div className="mb-1 font-medium text-fg">{labelFormat ? labelFormat(label) : label}</div>
      ) : null}
      <div className="flex flex-col gap-0.5">
        {payload.map((p) => (
          <div key={String(p.dataKey ?? p.name)} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-fg-secondary">
              <span aria-hidden className="size-2 rounded-[2px]" style={{ background: p.color }} />
              {p.name}
            </span>
            <span className="num font-mono text-fg">{p.value === undefined ? "—" : format(p.value, String(p.dataKey))}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-secondary">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-0.5 w-3.5 rounded"
            style={i.dashed ? { borderTop: `2px dashed ${i.color}`, height: 0 } : { background: i.color, height: 8, width: 8, borderRadius: 2 }}
          />
          {i.label}
        </span>
      ))}
    </div>
  );
}

export function ChartCard({
  title,
  description,
  provenance,
  legend,
  height = 260,
  loading,
  table,
  actions,
  children,
  className,
}: {
  title: string;
  description?: React.ReactNode;
  provenance?: readonly string[];
  legend?: React.ReactNode;
  height?: number;
  loading?: boolean;
  table?: { columns: string[]; rows: (string | number)[][] };
  actions?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  const [asTable, setAsTable] = React.useState(false);
  return (
    <figure className={cn("card flex min-w-0 flex-col", className)} aria-label={title}>
      <figcaption className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[13px] font-semibold text-fg">{title}</h3>
            <Provenance labels={provenance} />
          </div>
          {description ? <p className="mt-0.5 text-xs text-fg-muted">{description}</p> : null}
        </div>
        <div className="flex items-center gap-1">
          {actions}
          {table ? (
            <Button
              variant="ghost"
              size="icon"
              aria-label={asTable ? "Show chart" : "Show data table"}
              aria-pressed={asTable}
              onClick={() => setAsTable((v) => !v)}
            >
              {asTable ? <BarChart3 /> : <Table2 />}
            </Button>
          ) : null}
        </div>
      </figcaption>
      <div className="px-3 pb-3 pt-3">
        {legend ? <div className="mb-2 px-1">{legend}</div> : null}
        {loading ? (
          <ChartSkeleton height={height} />
        ) : asTable && table ? (
          <div className="max-h-[320px] overflow-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border text-left text-fg-muted">
                  {table.columns.map((c) => (
                    <th key={c} className="px-2 py-1.5 font-medium">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((r, i) => (
                  <tr key={i} className="border-b border-border/50">
                    {r.map((v, j) => (
                      <td key={j} className={cn("num px-2 py-1 font-mono", j > 0 && "text-right")}>
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ height }} className="w-full">
            {children}
          </div>
        )}
      </div>
    </figure>
  );
}

/** Loading placeholder sized like a chart. */
export function ChartSkeleton({ height = 260 }: { height?: number }) {
  return <div className="skeleton w-full" style={{ height }} />;
}
