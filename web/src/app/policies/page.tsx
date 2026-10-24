"use client";

import * as React from "react";
import { dump as yamlDump, load as yamlLoad } from "js-yaml";
import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, FlaskConical, Plus, Save, Trash2, Zap } from "lucide-react";
import { toast } from "sonner";
import { Dialog, PageHeader, Provenance, prov } from "@/components/ui/domain";
import { Badge, Button, Card, CardBody, CardHeader, EmptyState, ErrorState, Field, Input, Select, Tabs, TabsContent, TabsList, TabsTrigger, Textarea } from "@/components/ui/primitives";
import { api, unwrap } from "@/lib/api/client";
import { type SimulationOut, useActivePolicy, usePolicies, useSession } from "@/lib/api/hooks";
import { money, num, pct, pp } from "@/lib/format";
import { cn } from "@/lib/utils";

type Rule = { id: string; type: string; action: "flag" | "note"; reason: string; [k: string]: unknown };
type Policy = {
  version: string;
  name: string;
  description?: string;
  parent_version?: string | null;
  pd_threshold: { default: number | "model"; segments: { column: string; value: string; threshold: number }[] };
  band_cutoffs: number[];
  band_labels: string[];
  hard_rules: Rule[];
  cost_matrix: { mode: string; missed_default_cost: number; rejected_good_cost: number; review_cost: number; lost_margin_rate: number };
  economics: { lgd: number; ccf: number; annual_rate: number; funding_rate: number; opex_per_loan: number; horizon_years: number; default_exposure: number };
};
type Sim = Record<string, unknown> & {
  n: number;
  flag_rate: number;
  confusion: { tp: number; fp: number; tn: number; fn: number };
  recall: number | null;
  precision: number | null;
  costs: { total_cost: number };
};

function NumberField({ label, value, onChange, step = "any", suffix, provenance }: { label: string; value: number; onChange: (v: number) => void; step?: string; suffix?: string; provenance?: string[] }) {
  const [text, setText] = React.useState(String(value));
  const [seen, setSeen] = React.useState(value);
  if (value !== seen) {
    // the value changed from outside (e.g. the YAML view): show it (derived state, set during render)
    setSeen(value);
    setText(String(value));
  }
  return (
    <Field label={<>{label} {provenance ? <Provenance labels={provenance} /> : null}</>}>
      <div className="relative">
        <Input
          inputMode="decimal"
          value={text}
          step={step}
          onChange={(e) => {
            setText(e.target.value);
            const n = Number(e.target.value);
            if (e.target.value.trim() !== "" && Number.isFinite(n)) onChange(n);
          }}
          className="num pr-8 font-mono"
        />
        {suffix ? <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-[11px] text-fg-muted">{suffix}</span> : null}
      </div>
    </Field>
  );
}

const METRICS: { key: string; label: string; fmt: (v: unknown) => string; delta: (v: unknown) => string; prov: string }[] = [
  { key: "flag_rate", label: "Flag rate", fmt: (v) => pct(v), delta: (v) => pp(v), prov: "flag_rate" },
  { key: "recall", label: "Defaulters flagged (recall)", fmt: (v) => pct(v), delta: (v) => pp(v), prov: "recall" },
  { key: "precision", label: "Precision of flags", fmt: (v) => pct(v), delta: (v) => pp(v), prov: "precision" },
  { key: "default_rate_within_threshold", label: "Default rate within threshold", fmt: (v) => pct(v, 2), delta: (v) => pp(v, 2), prov: "default_rate_within_threshold" },
  { key: "ead_within_threshold", label: "EAD within threshold", fmt: (v) => money(v), delta: (v) => money(v), prov: "ead_within_threshold" },
  { key: "expected_loss_12m_within_threshold", label: "Expected loss within threshold", fmt: (v) => money(v), delta: (v) => money(v), prov: "expected_loss_12m_within_threshold" },
  { key: "expected_contribution_within_threshold", label: "Expected contribution", fmt: (v) => money(v), delta: (v) => money(v), prov: "expected_contribution_within_threshold" },
  { key: "realized_contribution_within_threshold", label: "Back-tested contribution", fmt: (v) => money(v), delta: (v) => money(v), prov: "realized_contribution_within_threshold" },
];

function Confusion({ c, title }: { c: Sim["confusion"]; title: string }) {
  const cells = [
    { k: "TP", v: c.tp, d: "Flagged, defaulted" },
    { k: "FP", v: c.fp, d: "Flagged, repaid" },
    { k: "FN", v: c.fn, d: "Within, defaulted" },
    { k: "TN", v: c.tn, d: "Within, repaid" },
  ];
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-fg-secondary">{title}</p>
      <div className="grid grid-cols-2 gap-1">
        {cells.map((x) => (
          <div key={x.k} className={cn("rounded-md border border-border px-2.5 py-2", (x.k === "TP" || x.k === "TN") ? "bg-surface-2" : "bg-surface")}>
            <div className="text-[10px] font-semibold uppercase tracking-wide text-fg-muted">{x.k}</div>
            <div className="num font-mono text-sm text-fg">{num(x.v)}</div>
            <div className="text-[10px] text-fg-muted">{x.d}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PoliciesPage() {
  const qc = useQueryClient();
  const active = useActivePolicy();
  const list = usePolicies();
  const session = useSession();
  const isAdmin = session.data?.role === "admin";
  const [draft, setDraft] = React.useState<Policy | null>(null);
  const [yamlText, setYamlText] = React.useState("");
  const [yamlError, setYamlError] = React.useState<string | null>(null);
  const [tab, setTab] = React.useState("form");
  const [split, setSplit] = React.useState<"validation" | "test">("validation");
  const [validation, setValidation] = React.useState<{ valid: boolean; errors: string[] } | null>(null);
  const [sim, setSim] = React.useState<SimulationOut | null>(null);
  const [simError, setSimError] = React.useState<unknown>(null);
  const [busy, setBusy] = React.useState(false);
  const [saveOpen, setSaveOpen] = React.useState(false);
  const [newVersion, setNewVersion] = React.useState("");

  const load = React.useCallback((p: Policy) => {
    setDraft(structuredClone(p));
    setYamlText(yamlDump(p, { lineWidth: 100, noRefs: true }));
    setYamlError(null);
  }, []);

  // Start from the active policy once it arrives (and again after "Reset" or a save clears the draft).
  if (active.data && draft === null) load(active.data.policy as unknown as Policy);

  const update = (fn: (p: Policy) => void) => {
    setDraft((d) => {
      if (!d) return d;
      const next = structuredClone(d);
      fn(next);
      setYamlText(yamlDump(next, { lineWidth: 100, noRefs: true }));
      return next;
    });
  };

  // Validate the draft with the API (debounced): the same rules as on save, including "model inputs only".
  React.useEffect(() => {
    if (!draft) return;
    const t = setTimeout(async () => {
      try {
        const r = await unwrap(api.POST("/v1/policies/validate", { body: { policy: draft as unknown as Record<string, never> } }));
        setValidation({ valid: r.valid, errors: r.errors });
      } catch {
        setValidation(null);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [draft]);

  const onYaml = (text: string) => {
    setYamlText(text);
    try {
      const parsed = yamlLoad(text);
      if (!parsed || typeof parsed !== "object") throw new Error("The document must be a mapping.");
      setDraft(parsed as Policy);
      setYamlError(null);
    } catch (e) {
      setYamlError((e as Error).message.split("\n")[0]);
    }
  };

  const simulate = async () => {
    if (!draft) return;
    setBusy(true);
    setSimError(null);
    try {
      setSim(await unwrap(api.POST("/v1/policies/simulate", { body: { policy: draft as unknown as Record<string, never>, split } })));
    } catch (e) {
      setSimError(e);
    } finally {
      setBusy(false);
    }
  };

  const save = async (activate: boolean) => {
    if (!draft) return;
    const body = { ...draft, version: newVersion.trim(), parent_version: active.data?.summary.version ?? null };
    try {
      await unwrap(api.POST("/v1/policies", { body: { policy: body as unknown as Record<string, never> } }));
      if (activate) await unwrap(api.POST("/v1/policies/{version}/activate", { params: { path: { version: body.version } } }));
      toast.success(activate ? `Policy ${body.version} saved and activated` : `Policy ${body.version} saved`);
      setSaveOpen(false);
      setDraft(null);
      await qc.invalidateQueries();
    } catch (e) {
      const err = e as { message: string; errors?: string[] };
      toast.error(err.errors?.length ? `${err.message}: ${err.errors[0]}` : err.message);
    }
  };

  const draftSim = sim?.draft as Sim | undefined;
  const activeSim = sim?.active as Sim | undefined;
  const p = sim?.provenance as Record<string, string[]> | undefined;

  return (
    <>
      <PageHeader
        title="Policy simulator"
        description="Lending policy is separate from the model. Edit a draft (form or YAML), simulate it on held-out applicants and compare it with the active policy. Rules may only use model inputs, never sensitive attributes."
        badges={active.data ? <Badge tone="accent" className="font-mono">active {active.data.summary.version}</Badge> : null}
        actions={
          <>
            <Select
              ariaLabel="Load a policy version"
              value={undefined}
              placeholder="Load version…"
              onValueChange={async (v) => {
                const r = await unwrap(api.GET("/v1/policies/{version}", { params: { path: { version: v } } }));
                load(r.policy as unknown as Policy);
                toast.message(`Loaded policy ${v} as the draft`);
              }}
              options={(list.data ?? []).map((x) => ({ value: x.version, label: `${x.version}${x.is_active ? " (active)" : ""}` }))}
              className="w-52"
            />
            <Button variant="secondary" onClick={() => active.data && load(active.data.policy as unknown as Policy)}>Reset to active</Button>
          </>
        }
      />
      {active.error ? <ErrorState error={active.error} className="mb-4" /> : null}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <Card className="self-start">
          <Tabs value={tab} onValueChange={setTab}>
            <CardHeader
              title="Draft policy"
              description={
                validation ? (
                  validation.valid ? (
                    <span className="inline-flex items-center gap-1"><CheckCircle2 className="size-3.5" style={{ color: "var(--good)" }} /> Valid</span>
                  ) : (
                    <span className="inline-flex items-center gap-1"><AlertTriangle className="size-3.5" style={{ color: "var(--critical)" }} /> {validation.errors.length} problem(s)</span>
                  )
                ) : (
                  "Validated as you edit"
                )
              }
              actions={
                <TabsList>
                  <TabsTrigger value="form">Form</TabsTrigger>
                  <TabsTrigger value="yaml">YAML</TabsTrigger>
                </TabsList>
              }
            />
            <CardBody>
              {!draft ? (
                <div className="skeleton h-64" />
              ) : (
                <>
                  {validation && !validation.valid ? (
                    <ErrorState className="mb-4" error={{ message: "The draft does not validate", errors: validation.errors }} />
                  ) : null}
                  <TabsContent value="form" className="flex flex-col gap-5">
                    <section className="grid grid-cols-2 gap-3">
                      <Field label="Name" className="col-span-2">
                        <Input value={draft.name} onChange={(e) => update((d) => void (d.name = e.target.value))} />
                      </Field>
                      <Field label={<>Default PD threshold <Provenance labels={["assumption"]} /></>}>
                        <Select
                          ariaLabel="Default threshold mode"
                          value={draft.pd_threshold.default === "model" ? "model" : "fixed"}
                          onValueChange={(v) => update((d) => void (d.pd_threshold.default = v === "model" ? "model" : 0.1))}
                          options={[{ value: "model", label: "Model's validation threshold" }, { value: "fixed", label: "Fixed value" }]}
                        />
                      </Field>
                      {draft.pd_threshold.default !== "model" ? (
                        <NumberField label="Fixed threshold" value={Number(draft.pd_threshold.default)} onChange={(v) => update((d) => void (d.pd_threshold.default = v))} />
                      ) : (
                        <div />
                      )}
                    </section>
                    <section>
                      <div className="mb-2 flex items-center justify-between">
                        <h4 className="text-xs font-semibold text-fg">Segment thresholds</h4>
                        <Button size="sm" variant="ghost" onClick={() => update((d) => d.pd_threshold.segments.push({ column: "NAME_CONTRACT_TYPE", value: "Cash loans", threshold: 0.15 }))}>
                          <Plus /> Add
                        </Button>
                      </div>
                      {draft.pd_threshold.segments.length === 0 ? <p className="text-xs text-fg-muted">None: every applicant uses the default threshold.</p> : null}
                      {draft.pd_threshold.segments.map((s, i) => (
                        <div key={i} className="mb-2 grid grid-cols-[1fr_1fr_6rem_auto] items-end gap-2">
                          <Field label="Column"><Input value={s.column} onChange={(e) => update((d) => void (d.pd_threshold.segments[i].column = e.target.value))} /></Field>
                          <Field label="Value"><Input value={s.value} onChange={(e) => update((d) => void (d.pd_threshold.segments[i].value = e.target.value))} /></Field>
                          <NumberField label="Threshold" value={s.threshold} onChange={(v) => update((d) => void (d.pd_threshold.segments[i].threshold = v))} />
                          <Button size="icon" variant="ghost" aria-label="Remove segment" onClick={() => update((d) => void d.pd_threshold.segments.splice(i, 1))}><Trash2 /></Button>
                        </div>
                      ))}
                    </section>
                    <section>
                      <h4 className="mb-2 text-xs font-semibold text-fg">Hard rules</h4>
                      <div className="flex flex-col gap-2">
                        {draft.hard_rules.map((r, i) => (
                          <div key={r.id + i} className="rounded-md border border-border p-3">
                            <div className="mb-2 flex flex-wrap items-center gap-2">
                              <span className="font-mono text-xs text-fg">{r.id}</span>
                              <Badge>{r.type}</Badge>
                              <div className="ml-auto w-32">
                                <Select ariaLabel={`${r.id} action`} value={r.action} onValueChange={(v) => update((d) => void (d.hard_rules[i].action = v as "flag" | "note"))} options={[{ value: "flag", label: "Flag" }, { value: "note", label: "Note only" }]} className="h-7 text-xs" />
                              </div>
                            </div>
                            <p className="mb-2 text-xs text-fg-muted">{r.reason}</p>
                            {r.type === "ratio_limit" ? (
                              <NumberField label={`Max ${String(r.numerator)} / ${String(r.denominator)}`} value={Number(r.max)} onChange={(v) => update((d) => void (d.hard_rules[i].max = v))} />
                            ) : r.type === "range" ? (
                              <div className="grid grid-cols-2 gap-2">
                                <NumberField label="Min" value={Number(r.min ?? 0)} onChange={(v) => update((d) => void (d.hard_rules[i].min = v))} />
                                <NumberField label="Max" value={Number(r.max ?? 0)} onChange={(v) => update((d) => void (d.hard_rules[i].max = v))} />
                              </div>
                            ) : (
                              <p className="text-xs text-fg-secondary">Column <span className="font-mono">{String(r.column)}</span>{r.type === "category_in" ? ` in ${(r.values as string[]).join(", ")}` : " is missing"}</p>
                            )}
                          </div>
                        ))}
                      </div>
                      <p className="mt-2 text-[11px] text-fg-muted">Add or remove rules in the YAML view.</p>
                    </section>
                    <section>
                      <h4 className="mb-2 text-xs font-semibold text-fg">Cost matrix and economics</h4>
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                        <NumberField label="LGD" value={draft.economics.lgd} onChange={(v) => update((d) => void (d.economics.lgd = v))} provenance={["assumption"]} />
                        <NumberField label="Lost margin (good flagged)" value={draft.cost_matrix.lost_margin_rate} onChange={(v) => update((d) => void (d.cost_matrix.lost_margin_rate = v))} provenance={["assumption"]} />
                        <NumberField label="Review cost / flag" value={draft.cost_matrix.review_cost} onChange={(v) => update((d) => void (d.cost_matrix.review_cost = v))} provenance={["assumption"]} />
                        <NumberField label="Lending rate" value={draft.economics.annual_rate} onChange={(v) => update((d) => void (d.economics.annual_rate = v))} provenance={["assumption"]} />
                        <NumberField label="Funding rate" value={draft.economics.funding_rate} onChange={(v) => update((d) => void (d.economics.funding_rate = v))} provenance={["assumption"]} />
                        <NumberField label="Opex / loan / year" value={draft.economics.opex_per_loan} onChange={(v) => update((d) => void (d.economics.opex_per_loan = v))} provenance={["assumption"]} />
                      </div>
                    </section>
                  </TabsContent>
                  <TabsContent value="yaml">
                    <Textarea value={yamlText} onChange={(e) => onYaml(e.target.value)} rows={28} spellCheck={false} aria-label="Policy YAML" aria-invalid={Boolean(yamlError)} />
                    {yamlError ? <p className="mt-2 font-mono text-[11px] text-critical">{yamlError}</p> : null}
                  </TabsContent>
                </>
              )}
            </CardBody>
          </Tabs>
          <div className="flex flex-wrap items-center gap-2 border-t border-border px-4 py-3">
            <div className="w-44">
              <Select
                ariaLabel="Simulation split"
                value={split}
                onValueChange={(v) => setSplit(v as "validation" | "test")}
                options={[{ value: "validation", label: "Validation (tune here)" }, { value: "test", label: "Test (read-only check)" }]}
              />
            </div>
            <Button variant="primary" onClick={simulate} disabled={!draft || busy || Boolean(yamlError)} data-testid="simulate">
              <FlaskConical /> {busy ? "Simulating…" : "Simulate"}
            </Button>
            {isAdmin ? (
              <Button variant="secondary" disabled={!validation?.valid} onClick={() => { setNewVersion(""); setSaveOpen(true); }}>
                <Save /> Save as new version
              </Button>
            ) : (
              <span className="text-[11px] text-fg-muted">Saving needs an admin key.</span>
            )}
          </div>
        </Card>

        <div className="flex min-w-0 flex-col gap-4">
          {simError ? <ErrorState error={simError} /> : null}
          {sim && draftSim && activeSim ? (
            <>
              {sim.warnings.length ? (
                <div className="rounded-md border border-warning/40 bg-warning-soft px-3 py-2 text-xs text-fg">
                  {sim.warnings.map((w) => (
                    <p key={w} className="flex gap-1.5"><AlertTriangle className="mt-0.5 size-3.5 shrink-0" style={{ color: "var(--warning)" }} />{w}</p>
                  ))}
                </div>
              ) : null}
              <Card data-testid="simulation-result">
                <CardHeader title="Draft vs active" description={`${num(draftSim.n)} ${sim.split} applicants · model v${sim.model_version}`} />
                <div className="overflow-x-auto">
                  <table className="w-full text-[13px]">
                    <thead>
                      <tr className="border-b border-border text-[11px] uppercase tracking-wide text-fg-muted">
                        <th className="px-4 py-2 text-left font-medium">Metric</th>
                        <th className="px-4 py-2 text-right font-medium">Active</th>
                        <th className="px-4 py-2 text-right font-medium">Draft</th>
                        <th className="px-4 py-2 text-right font-medium">Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {METRICS.map((m) => {
                        const d = (sim.delta as Record<string, number | null>)[m.key];
                        return (
                          <tr key={m.key} className="border-b border-border/60">
                            <td className="px-4 py-2 text-fg-secondary"><span className="flex flex-wrap items-center gap-1.5">{m.label} <Provenance labels={prov(p, m.prov)} /></span></td>
                            <td className="num px-4 py-2 text-right font-mono text-xs">{m.fmt(activeSim[m.key])}</td>
                            <td className="num px-4 py-2 text-right font-mono text-xs text-fg">{m.fmt(draftSim[m.key])}</td>
                            <td className="num px-4 py-2 text-right font-mono text-xs" style={{ color: d ? "var(--fg)" : "var(--fg-muted)" }}>{d === null || d === undefined ? "—" : m.delta(d)}</td>
                          </tr>
                        );
                      })}
                      <tr>
                        <td className="px-4 py-2 text-fg-secondary"><span className="flex items-center gap-1.5">Total decision cost <Provenance labels={prov(p, "total_cost")} /></span></td>
                        <td className="num px-4 py-2 text-right font-mono text-xs">{money(activeSim.costs.total_cost)}</td>
                        <td className="num px-4 py-2 text-right font-mono text-xs text-fg">{money(draftSim.costs.total_cost)}</td>
                        <td className="num px-4 py-2 text-right font-mono text-xs">{money((sim.delta as Record<string, number>).total_cost)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
              <Card>
                <CardHeader title="Confusion matrices" description="Flagged vs observed outcome" />
                <CardBody className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <Confusion c={activeSim.confusion} title="Active" />
                  <Confusion c={draftSim.confusion} title="Draft" />
                </CardBody>
              </Card>
            </>
          ) : (
            <Card>
              <EmptyState icon={<Zap />} title="Run a simulation" body="Change a threshold or rule, then press Simulate. The draft and the active policy run over the same held-out applicants; the validation split is the default because that is where choices are made." />
            </Card>
          )}
        </div>
      </div>
      <Dialog open={saveOpen} onOpenChange={setSaveOpen} title="Save as a new policy version" description="Versions are immutable. The draft is validated again on the server.">
        <div className="flex flex-col gap-4">
          <Field label="New version" hint={`Parent: ${active.data?.summary.version ?? "—"}`}>
            <Input value={newVersion} onChange={(e) => setNewVersion(e.target.value)} placeholder="e.g. 2026.11.0" autoFocus />
          </Field>
          <div className="flex justify-end gap-2">
            <Button variant="secondary" disabled={!newVersion.trim()} onClick={() => save(false)}>Save</Button>
            <Button variant="primary" disabled={!newVersion.trim()} onClick={() => save(true)}>Save and activate</Button>
          </div>
        </div>
      </Dialog>
    </>
  );
}
