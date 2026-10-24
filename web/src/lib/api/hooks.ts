"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "./client";
import type { components } from "./schema";

/** Response types generated from the API's OpenAPI document (`pnpm gen:api`). */
export type Schemas = components["schemas"];
export type ScoreResponse = Schemas["ScoreResponse"];
export type DecideResponse = Schemas["DecideResponse"];
export type MetaOut = Schemas["MetaOut"];
export type ModelList = Schemas["ModelList"];
export type ModelSchemaOut = Schemas["ModelSchemaOut"];
export type FieldSpec = Schemas["FieldSpec"];
export type TableOut = Schemas["TableOut"];
export type ResultOut = Schemas["ResultOut"];
export type SimulationOut = Schemas["SimulationOut"];
export type PolicyDocument = Schemas["PolicyDocument"];
export type PolicySummary = Schemas["PolicySummary"];
export type ProvenanceMap = Record<string, string[]>;
type Split = "validation" | "test";

export const keys = {
  meta: ["meta"] as const,
  session: ["session"] as const,
  overview: ["overview"] as const,
  models: ["models"] as const,
  model: (v: string) => ["model", v] as const,
};

export function useSession() {
  return useQuery({
    queryKey: keys.session,
    queryFn: async () => (await fetch("/api/session")).json() as Promise<{ role: "analyst" | "admin"; adminAvailable: boolean }>,
    staleTime: 60_000,
  });
}

export function useSetRole() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (role: "analyst" | "admin") =>
      (await fetch("/api/session", { method: "POST", body: JSON.stringify({ role }), headers: { "content-type": "application/json" } })).json(),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export function useMeta() {
  return useQuery({ queryKey: keys.meta, queryFn: () => unwrap(api.GET("/v1/meta")), staleTime: 15_000, refetchInterval: 30_000 });
}

export function useOverview() {
  return useQuery({ queryKey: keys.overview, queryFn: () => unwrap(api.GET("/v1/overview", { params: { query: { days: 14 } } })) });
}

export function useModels() {
  return useQuery({ queryKey: keys.models, queryFn: () => unwrap(api.GET("/v1/models")) });
}

export function useModel(version: string | undefined) {
  return useQuery({
    queryKey: ["model", version],
    enabled: Boolean(version),
    queryFn: () => unwrap(api.GET("/v1/models/{version}", { params: { path: { version: version! } } })),
  });
}

export function useModelSchema(version: string | undefined) {
  return useQuery({
    queryKey: ["model-schema", version],
    enabled: Boolean(version),
    staleTime: Infinity,
    queryFn: () => unwrap(api.GET("/v1/models/{version}/schema", { params: { path: { version: version! } } })),
  });
}

export function useEvaluation(version: string | undefined) {
  return useQuery({
    queryKey: ["model-eval", version],
    enabled: Boolean(version),
    staleTime: Infinity,
    queryFn: () => unwrap(api.GET("/v1/models/{version}/evaluation", { params: { path: { version: version! } } })),
  });
}

export function useModelCard(version: string | undefined) {
  return useQuery({
    queryKey: ["model-card", version],
    enabled: Boolean(version),
    staleTime: Infinity,
    queryFn: () => unwrap(api.GET("/v1/models/{version}/card", { params: { path: { version: version! } } })),
  });
}

export function usePromote() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (version: string) =>
      unwrap(api.POST("/v1/models/{version}/promote", { params: { path: { version } }, body: { alias: "production" } })),
    onSuccess: () => qc.invalidateQueries(),
  });
}

export type PortfolioView = "summary" | "bands" | "deciles" | "segments" | "concentration";

export function usePortfolio(view: PortfolioView, q: { split: Split; lgd?: number; ccf?: number; column?: string }) {
  return useQuery({
    queryKey: ["portfolio", view, q],
    placeholderData: keepPreviousData,
    queryFn: (): Promise<TableOut> => {
      const params = { params: { query: q } };
      switch (view) {
        case "summary":
          return unwrap(api.GET("/v1/portfolio/summary", params));
        case "bands":
          return unwrap(api.GET("/v1/portfolio/bands", params));
        case "deciles":
          return unwrap(api.GET("/v1/portfolio/deciles", params));
        case "segments":
          return unwrap(api.GET("/v1/portfolio/segments", params));
        default:
          return unwrap(api.GET("/v1/portfolio/concentration", params));
      }
    },
  });
}

export function useFairness(q: { attribute?: string; split: Split }) {
  return useQuery({
    queryKey: ["fairness", q],
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/v1/fairness/report", { params: { query: q } })),
  });
}

export function useDrift() {
  return useQuery({ queryKey: ["drift"], queryFn: () => unwrap(api.GET("/v1/monitoring/drift")) });
}

export function useDriftHistory() {
  return useQuery({ queryKey: ["drift-history"], queryFn: () => unwrap(api.GET("/v1/monitoring/history", { params: { query: { limit: 50 } } })) });
}

export function useAlerts() {
  return useQuery({ queryKey: ["alerts"], queryFn: () => unwrap(api.GET("/v1/monitoring/alerts", { params: { query: { limit: 100 } } })) });
}

export function useDecisions(q: { q?: string; outcome?: "within_threshold" | "flagged_for_review"; limit: number }) {
  return useQuery({
    queryKey: ["decisions", q],
    placeholderData: keepPreviousData,
    queryFn: () => unwrap(api.GET("/v1/decisions", { params: { query: q } })),
  });
}

export function usePolicies() {
  return useQuery({ queryKey: ["policies"], queryFn: () => unwrap(api.GET("/v1/policies")) });
}

export function useActivePolicy() {
  return useQuery({ queryKey: ["policy-active"], queryFn: () => unwrap(api.GET("/v1/policies/active")) });
}
