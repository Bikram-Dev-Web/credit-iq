import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { ROLE_COOKIE, adminAvailable, currentRole } from "@/lib/api/server";

/** The viewer's role (analyst / admin) — a convenience switch for the demo; the API enforces roles by key. */
export async function GET() {
  return Response.json({ role: await currentRole(), adminAvailable: adminAvailable() });
}

export async function POST(req: NextRequest) {
  const body = (await req.json().catch(() => ({}))) as { role?: string };
  const role = body.role === "admin" && adminAvailable() ? "admin" : "analyst";
  const jar = await cookies();
  jar.set(ROLE_COOKIE, role, { httpOnly: true, sameSite: "lax", path: "/", maxAge: 60 * 60 * 24 * 30 });
  return Response.json({ role, adminAvailable: adminAvailable() });
}
