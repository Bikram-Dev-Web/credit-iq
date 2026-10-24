import "server-only";
import { cookies } from "next/headers";

export type Role = "analyst" | "admin";
export const ROLE_COOKIE = "ciq_role";

export function apiUrl(): string {
  return (process.env.CREDITIQ_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
}

export function adminAvailable(): boolean {
  return Boolean(process.env.CREDITIQ_ADMIN_KEY);
}

export async function currentRole(): Promise<Role> {
  const jar = await cookies();
  return jar.get(ROLE_COOKIE)?.value === "admin" && adminAvailable() ? "admin" : "analyst";
}

export async function apiKeyFor(role: Role): Promise<string | undefined> {
  return role === "admin" ? process.env.CREDITIQ_ADMIN_KEY : process.env.CREDITIQ_ANALYST_KEY;
}
