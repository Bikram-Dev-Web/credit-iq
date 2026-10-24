import type { Metadata } from "next";
import { PageHeader } from "@/components/ui/domain";

export const metadata: Metadata = { title: "Docs & about" };

type Box = { id: string; x: number; y: number; w: number; label: string; sub: string; accent?: boolean };
const BOXES: Box[] = [
  { id: "web", x: 20, y: 40, w: 150, label: "Web dashboard", sub: "Next.js · server proxy" },
  { id: "api", x: 230, y: 40, w: 170, label: "Scoring API", sub: "FastAPI · auth · metrics", accent: true },
  { id: "policy", x: 460, y: 40, w: 150, label: "Policy engine", sub: "versioned YAML rules" },
  { id: "pg", x: 670, y: 40, w: 150, label: "Postgres", sub: "audit log · policies · drift" },
  { id: "trainer", x: 230, y: 190, w: 170, label: "Trainer", sub: "creditiq-train · YAML" },
  { id: "mlflow", x: 460, y: 190, w: 150, label: "MLflow registry", sub: "staging · production" },
  { id: "monitor", x: 670, y: 190, w: 150, label: "Monitor job", sub: "PSI · alerts · schedule" },
  { id: "prom", x: 20, y: 190, w: 150, label: "Prometheus + Grafana", sub: "latency · PD · drift" },
];
const EDGES: [string, string, string][] = [
  ["web", "api", "HTTPS"],
  ["api", "policy", "decide"],
  ["policy", "pg", "audit"],
  ["trainer", "mlflow", "register"],
  ["mlflow", "api", "load @production"],
  ["monitor", "pg", "runs, alerts"],
  ["api", "prom", "/metrics"],
];

function center(b: Box) {
  return { x: b.x + b.w / 2, y: b.y + 30 };
}

function Architecture() {
  const byId = Object.fromEntries(BOXES.map((b) => [b.id, b]));
  return (
    <svg viewBox="0 0 840 280" className="w-full min-w-[640px]" role="img" aria-labelledby="arch-title">
      <title id="arch-title">CreditIQ architecture: web dashboard, scoring API, policy engine, Postgres, trainer, MLflow registry, monitor job, Prometheus and Grafana</title>
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0 0 10 5 0 10z" fill="var(--fg-muted)" />
        </marker>
      </defs>
      {EDGES.map(([a, b, label]) => {
        const s = center(byId[a]);
        const t = center(byId[b]);
        const horizontal = Math.abs(s.y - t.y) < 5;
        const x1 = horizontal ? (s.x < t.x ? byId[a].x + byId[a].w : byId[a].x) : s.x;
        const x2 = horizontal ? (s.x < t.x ? byId[b].x : byId[b].x + byId[b].w) : t.x;
        const y1 = horizontal ? s.y : s.y < t.y ? byId[a].y + 60 : byId[a].y;
        const y2 = horizontal ? t.y : s.y < t.y ? byId[b].y : byId[b].y + 60;
        const diag = !horizontal && Math.abs(x1 - x2) > 5;
        const path = diag ? `M${x1} ${y1} C ${x1} ${(y1 + y2) / 2}, ${x2} ${(y1 + y2) / 2}, ${x2} ${y2}` : `M${x1} ${y1} L${x2} ${y2}`;
        return (
          <g key={a + b}>
            <path d={path} stroke="var(--border-strong)" strokeWidth={1.5} fill="none" markerEnd="url(#arrow)" />
            <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fontSize="10" fill="var(--fg-muted)">
              {label}
            </text>
          </g>
        );
      })}
      {BOXES.map((b) => (
        <g key={b.id}>
          <rect x={b.x} y={b.y} width={b.w} height={60} rx={8} fill="var(--surface-2)" stroke={b.accent ? "var(--accent)" : "var(--border-strong)"} strokeWidth={b.accent ? 1.5 : 1} />
          <text x={b.x + 12} y={b.y + 26} fontSize="13" fontWeight={600} fill="var(--fg)">{b.label}</text>
          <text x={b.x + 12} y={b.y + 44} fontSize="11" fill="var(--fg-muted)">{b.sub}</text>
        </g>
      ))}
      <text x={420} y={272} textAnchor="middle" fontSize="10" fill="var(--fg-muted)">
        All Python services share creditiq_core (features, models, calibration, finance, fairness, monitoring).
      </text>
    </svg>
  );
}

const SECTIONS: { title: string; items: string[] }[] = [
  {
    title: "Methodology",
    items: [
      "Three model families (logistic regression, random forest, LightGBM), trained on a training split, calibrated (Platt) and thresholded on a validation split, and reported once on an untouched test split.",
      "The leakage guard screens predictors on training rows only; the target mapping is always explicit — 1 is never assumed to mean default.",
      "Sensitive attributes (gender, age, family status) are never model inputs; they are kept only for the fairness audit and can never be used by a policy rule.",
      "Expected loss = PD × LGD × EAD (12-month). Contribution, break-even, stress and NPV use one horizon convention throughout.",
      "Lending policy is a separate, versioned document: thresholds by segment, hard rules (flag or note), band cutoffs and a cost matrix. Policies are simulated on validation by default.",
    ],
  },
  {
    title: "Provenance labels",
    items: [
      "Observed — historical facts (labels, counts, exposures). Measured — evaluation on held-out data.",
      "Model — the model's own output. Assumption — a value a person chose (LGD, CCF, rates, thresholds).",
      "Simulated — projections combining model output and assumptions. Synthetic — computed on fabricated data.",
    ],
  },
  {
    title: "Privacy",
    items: [
      "The audit log stores scores, outcomes, rules fired, explanation contributions and an HMAC-SHA256 of the inputs — never the inputs.",
      "API keys stay on the server: the browser talks to a Next.js proxy, which adds the key.",
      "Drift monitoring compares batches you provide with binned training statistics; live PD drift uses the scores in the audit log.",
    ],
  },
  {
    title: "Limitations",
    items: [
      "The bundled data is synthetic: results demonstrate the workflow, not real-world performance.",
      "PD is the probability of the dataset's own event (for Home Credit, an early payment difficulty); treating it as a 12-month PD is an assumption.",
      "LGD, EAD, rates and costs are assumptions; EL and contribution inherit their uncertainty. This is not an IFRS 9 or capital model.",
      "Explanations show association, not causation. Fairness checks cover only the attributes available and are not legal advice.",
      "Random splits do not test performance on future cohorts.",
    ],
  },
];

export default function DocsPage() {
  return (
    <>
      <PageHeader title="Docs & about" description="How CreditIQ is built, what its numbers mean, and what it must not be used for." />
      <section className="card mb-4 overflow-x-auto p-4" aria-label="Architecture">
        <h2 className="mb-3 text-[13px] font-semibold text-fg">Architecture</h2>
        <Architecture />
      </section>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {SECTIONS.map((s) => (
          <section key={s.title} className="card p-4">
            <h2 className="mb-2 text-[13px] font-semibold text-fg">{s.title}</h2>
            <ul className="flex list-disc flex-col gap-1.5 pl-4 text-[13px] text-fg-secondary">
              {s.items.map((i) => (
                <li key={i}>{i}</li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <section className="mt-4 rounded-lg border border-serious/40 bg-serious-soft p-4 text-[13px] text-fg">
        <strong>Educational decision-support prototype, not for real credit decisions.</strong> CreditIQ does not approve, decline or price credit for real
        people. Any real use would need independent model validation, legal review of inputs and reasons, documented human review of flagged applications,
        and monitoring with defined escalation.
      </section>
    </>
  );
}
