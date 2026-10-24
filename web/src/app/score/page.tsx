"use client";

import * as React from "react";
import * as Collapsible from "@radix-ui/react-collapsible";
import { ChevronRight, RotateCcw, Sparkles, Wand2 } from "lucide-react";
import { toast } from "sonner";
import { z } from "zod";
import { PageHeader } from "@/components/ui/domain";
import { Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Field, Input, Select, Skeleton, Slider, Switch } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { type DecideResponse, type FieldSpec, type ScoreResponse, useActivePolicy, useMeta, useModelSchema } from "@/lib/api/hooks";
import { pct } from "@/lib/format";
import { cn } from "@/lib/utils";
import { ResultPanel } from "./result";

type Values = Record<string, string>;
const UNKNOWN = "__unknown__";

function defaults(fields: FieldSpec[]): Values {
  const v: Values = {};
  for (const f of fields) {
    if (f.kind === "numeric") {
      const m = f.median ?? 0;
      v[f.name] = String(f.is_integer ? Math.round(m) : Number(m.toPrecision(6)));
    } else v[f.name] = f.mode ?? "";
  }
  return v;
}

/** Client-side validation mirrors the API's generated schema: numbers must be finite; blank means unknown. */
function buildSchema(fields: FieldSpec[]) {
  const shape: Record<string, z.ZodType<string>> = {};
  for (const f of fields) {
    shape[f.name] =
      f.kind === "numeric"
        ? z.string().refine((s) => s.trim() === "" || Number.isFinite(Number(s)), { message: "Enter a number, or leave blank for unknown" })
        : z.string().max(200);
  }
  return z.object(shape);
}

function toPayload(fields: FieldSpec[], values: Values): Record<string, number | string | null> {
  const out: Record<string, number | string | null> = {};
  for (const f of fields) {
    const raw = (values[f.name] ?? "").trim();
    out[f.name] = raw === "" || raw === UNKNOWN ? null : f.kind === "numeric" ? Number(raw) : raw;
  }
  return out;
}

function FieldInput({ f, value, onChange, error }: { f: FieldSpec; value: string; onChange: (v: string) => void; error?: string }) {
  const id = `f-${f.name}`;
  const hint =
    f.kind === "numeric"
      ? `typical ${Number(f.p01 ?? 0).toLocaleString()} – ${Number(f.p99 ?? 0).toLocaleString()}`
      : `${(f.categories ?? []).length} categories`;
  return (
    <Field label={f.name} htmlFor={id} hint={hint} error={error}>
      {f.kind === "numeric" ? (
        <Input id={id} inputMode="decimal" value={value} aria-invalid={Boolean(error)} onChange={(e) => onChange(e.target.value)} placeholder="unknown" className="num font-mono" />
      ) : (
        <Select
          id={id}
          ariaLabel={f.name}
          value={value === "" ? UNKNOWN : value}
          onValueChange={(v) => onChange(v === UNKNOWN ? "" : v)}
          options={[{ value: UNKNOWN, label: "Unknown" }, ...(f.categories ?? []).map((c) => ({ value: c, label: c }))]}
        />
      )}
    </Field>
  );
}

export default function ScorePage() {
  const meta = useMeta();
  const version = meta.data?.model_version ?? undefined;
  const schema = useModelSchema(version);
  const policy = useActivePolicy();
  const fields = React.useMemo(() => schema.data?.fields ?? [], [schema.data]);
  const [values, setValues] = React.useState<Values>({});
  const [errors, setErrors] = React.useState<Record<string, string>>({});
  const [result, setResult] = React.useState<ScoreResponse | DecideResponse | null>(null);
  const [error, setError] = React.useState<unknown>(null);
  const [busy, setBusy] = React.useState(false);
  const [whatIf, setWhatIf] = React.useState(false);
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const cutoffs = (policy.data?.policy?.band_cutoffs as number[] | undefined) ?? [0.05, 0.1, 0.2];

  // New schema (first load or a different model version): pre-fill typical values.
  const [schemaOf, setSchemaOf] = React.useState<string | undefined>(undefined);
  if (fields.length && schema.data && schemaOf !== schema.data.model_version) {
    setSchemaOf(schema.data.model_version);
    setValues(defaults(fields));
  }

  const validate = React.useCallback(() => {
    const parsed = buildSchema(fields).safeParse(values);
    if (parsed.success) {
      setErrors({});
      return true;
    }
    const e: Record<string, string> = {};
    for (const issue of parsed.error.issues) e[String(issue.path[0])] = issue.message;
    setErrors(e);
    return false;
  }, [fields, values]);

  const submit = React.useCallback(
    async (mode: "decide" | "score") => {
      if (!validate()) return;
      setBusy(true);
      setError(null);
      try {
        // Applicant fields come from the served model's schema at runtime (the API validates them), so the
        // generated static type for one model version is not used here.
        const body = { applicant: toPayload(fields, values) } as unknown as { applicant: never };
        const r =
          mode === "decide"
            ? await unwrap(api.POST("/v1/decide", { body }))
            : await unwrap(api.POST("/v1/score", { body }));
        setResult(r);
      } catch (e) {
        setError(e);
        if (mode === "decide") toast.error("Scoring failed");
      } finally {
        setBusy(false);
      }
    },
    [fields, values, validate],
  );

  // What-if: re-score (without hard rules) shortly after the inputs stop changing.
  const first = React.useRef(true);
  React.useEffect(() => {
    if (!whatIf || !fields.length) return;
    if (first.current) {
      first.current = false;
      return;
    }
    const t = setTimeout(() => void submit("score"), 350);
    return () => clearTimeout(t);
  }, [values, whatIf, fields.length, submit]);

  const main = fields.filter((f) => !f.advanced);
  const advanced = fields.filter((f) => f.advanced);
  const sliders = main.filter((f) => f.kind === "numeric" && f.p01 !== null && f.p99 !== null && f.p99! > f.p01!).slice(0, 6);
  const set = (name: string, v: string) => setValues((s) => ({ ...s, [name]: v }));

  return (
    <>
      <PageHeader
        title="Score an applicant"
        description="Enter the model's inputs (most influential first). Blank means unknown: the model imputes it from training data. The outcome is within threshold or flagged for review — never an approval or a decline."
      />
      <div className="mb-4 rounded-md border border-border bg-surface-2 px-3 py-2 text-xs text-fg-secondary">
        Educational decision-support prototype, not for real credit decisions. The served model is trained on{" "}
        {meta.data?.data_provenance === "synthetic" ? "synthetic (fabricated) applicants" : "historical data"}.
      </div>
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <Card className="self-start">
          <CardHeader
            title="Applicant inputs"
            description={version ? `Schema of model v${version}` : "Loading model schema…"}
            actions={
              <>
                <label className="flex items-center gap-2 text-xs text-fg-secondary">
                  <Switch checked={whatIf} onCheckedChange={(v) => { first.current = true; setWhatIf(v); if (v) void submit("score"); }} aria-label="What-if mode" />
                  What-if
                </label>
                <Button variant="ghost" size="sm" onClick={() => setValues(defaults(fields))} aria-label="Reset inputs to typical values">
                  <RotateCcw /> Reset
                </Button>
              </>
            }
          />
          <CardBody>
            {schema.isLoading || meta.isLoading ? (
              <div className="grid grid-cols-2 gap-3">
                {Array.from({ length: 10 }).map((_, i) => (
                  <Skeleton key={i} className="h-12" />
                ))}
              </div>
            ) : schema.error || meta.error ? (
              <ErrorState error={schema.error ?? meta.error} />
            ) : !fields.length ? (
              <EmptyState title="No model is being served" body="Train and promote a model (creditiq-train), then reload." />
            ) : (
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void submit("decide");
                }}
                className="flex flex-col gap-4"
                aria-label="Applicant inputs"
              >
                {whatIf && sliders.length ? (
                  <div className="flex flex-col gap-3 rounded-md border border-accent/30 bg-accent-soft p-3">
                    <p className="flex items-center gap-1.5 text-xs font-medium text-fg">
                      <Wand2 className="size-3.5 text-accent" /> Drag to re-score live (debounced; hard rules apply on Assess)
                    </p>
                    {sliders.map((f) => {
                      const lo = Number(f.p01);
                      const hi = Number(f.p99);
                      const step = f.is_integer ? 1 : (hi - lo) / 200;
                      const v = Number(values[f.name]);
                      return (
                        <div key={f.name} className="grid grid-cols-[minmax(0,10rem)_1fr_5.5rem] items-center gap-3">
                          <span className="truncate text-xs text-fg-secondary">{f.name}</span>
                          <Slider
                            aria-label={`${f.name} slider`}
                            min={lo}
                            max={hi}
                            step={step}
                            value={[Number.isFinite(v) ? Math.min(Math.max(v, lo), hi) : lo]}
                            onValueChange={([x]) => set(f.name, String(f.is_integer ? Math.round(x) : Number(x.toPrecision(5))))}
                          />
                          <span className="num text-right font-mono text-xs text-fg">{values[f.name] || "—"}</span>
                        </div>
                      );
                    })}
                  </div>
                ) : null}
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {main.map((f) => (
                    <FieldInput key={f.name} f={f} value={values[f.name] ?? ""} onChange={(v) => set(f.name, v)} error={errors[f.name]} />
                  ))}
                </div>
                {advanced.length ? (
                  <Collapsible.Root open={advancedOpen} onOpenChange={setAdvancedOpen}>
                    <Collapsible.Trigger className="flex items-center gap-1 text-xs font-medium text-fg-secondary hover:text-fg">
                      <ChevronRight className={cn("size-3.5 transition-transform duration-150", advancedOpen && "rotate-90")} />
                      Advanced inputs ({advanced.length})
                    </Collapsible.Trigger>
                    <Collapsible.Content className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
                      {advanced.map((f) => (
                        <FieldInput key={f.name} f={f} value={values[f.name] ?? ""} onChange={(v) => set(f.name, v)} error={errors[f.name]} />
                      ))}
                    </Collapsible.Content>
                  </Collapsible.Root>
                ) : null}
                <div className="flex items-center gap-2 border-t border-border pt-4">
                  <Button type="submit" variant="primary" disabled={busy} data-testid="assess">
                    <Sparkles /> {busy ? "Assessing…" : "Assess applicant"}
                  </Button>
                  {result ? <span className="text-xs text-fg-muted">last PD {pct(result.pd)} · {result.latency_ms.toFixed(1)} ms</span> : null}
                </div>
              </form>
            )}
          </CardBody>
        </Card>
        <div className="min-w-0">
          {error ? <ErrorState error={error} className="mb-4" /> : null}
          {result ? (
            <ResultPanel r={result} cutoffs={cutoffs} />
          ) : (
            <Card>
              <EmptyState icon={<Sparkles />} title="No assessment yet" body="Fill in the inputs (typical values are pre-filled) and press Assess applicant. Turn on What-if to see the PD move as you drag the most influential inputs." />
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
