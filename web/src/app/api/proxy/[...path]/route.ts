import { NextRequest } from "next/server";
import { apiKeyFor, apiUrl, currentRole } from "@/lib/api/server";

/**
 * Server-side proxy to the CreditIQ API. The API key (chosen by the role cookie) is added here, so it never
 * reaches the browser. Only the API's own paths are forwarded.
 */
async function forward(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const target = path.join("/");
  if (!/^(v1\/|healthz$|readyz$)/.test(target)) {
    return Response.json({ error: { message: "Not an API path" } }, { status: 404 });
  }
  const role = await currentRole();
  const key = await apiKeyFor(role);
  const headers = new Headers();
  for (const h of ["content-type", "accept", "x-request-id"]) {
    const v = req.headers.get(h);
    if (v) headers.set(h, v);
  }
  if (key) headers.set("x-api-key", key);
  const init: RequestInit & { duplex?: string } = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = req.body;
    init.duplex = "half";
  }
  let res: Response;
  try {
    res = await fetch(`${apiUrl()}/${target}${req.nextUrl.search}`, init);
  } catch {
    return Response.json(
      { error: { message: "The CreditIQ API is unreachable. Is it running (CREDITIQ_API_URL)?" } },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const h of ["content-type", "x-request-id", "x-batch-id", "x-model-version", "x-policy-version", "retry-after"]) {
    const v = res.headers.get(h);
    if (v) out.set(h, v);
  }
  return new Response(res.body, { status: res.status, headers: out });
}

export const GET = forward;
export const POST = forward;
export const dynamic = "force-dynamic";
