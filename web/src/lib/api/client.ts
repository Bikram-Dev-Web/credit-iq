import createClient from "openapi-fetch";
import type { paths } from "./schema";

/** Typed API client. Requests go to the Next.js server proxy, which adds the API key server-side. */
export const api = createClient<paths>({ baseUrl: "/api/proxy" });

export class ApiError extends Error {
  status: number;
  errors: string[];
  constructor(status: number, message: string, errors: string[] = []) {
    super(message);
    this.status = status;
    this.errors = errors;
  }
}

type ErrorBody = { error?: { message?: string; errors?: unknown[] } | string; detail?: unknown };

export function toApiError(status: number, body: unknown): ApiError {
  const b = (body ?? {}) as ErrorBody;
  const err = b.error;
  if (typeof err === "string") return new ApiError(status, err);
  const list = (err?.errors ?? []).map((e) => {
    if (typeof e === "string") return e;
    const o = e as { loc?: unknown[]; msg?: string };
    return `${(o.loc ?? []).filter((x) => x !== "body").join(".")}: ${o.msg ?? "invalid"}`;
  });
  return new ApiError(status, err?.message ?? `Request failed (${status})`, list);
}

/** Unwrap an openapi-fetch result or throw an ApiError. */
export async function unwrap<T>(p: Promise<{ data?: T; error?: unknown; response: Response }>): Promise<T> {
  const { data, error, response } = await p;
  if (error !== undefined || !response.ok) throw toApiError(response.status, error);
  return data as T;
}
