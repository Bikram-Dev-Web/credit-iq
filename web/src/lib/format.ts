/** Number formatting. Every rendered number goes through these (tabular figures are applied by CSS). */

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function pct(v: unknown, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : "—";
}

export function num(v: unknown, digits = 0): string {
  return isNum(v) ? v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "—";
}

export function dec(v: unknown, digits = 3): string {
  return isNum(v) ? v.toFixed(digits) : "—";
}

export function money(v: unknown, compact = true): string {
  if (!isNum(v)) return "—";
  const sign = v < 0 ? "−" : "";
  const a = Math.abs(v);
  if (compact) {
    for (const [div, suf] of [
      [1e12, "T"],
      [1e9, "B"],
      [1e6, "M"],
      [1e3, "K"],
    ] as const) {
      if (a >= div) return `${sign}${(a / div).toFixed(2)}${suf}`;
    }
  }
  return `${sign}${a.toLocaleString("en-US", { maximumFractionDigits: 0 })}`;
}

export function signed(v: unknown, digits = 3): string {
  return isNum(v) ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}` : "—";
}

export function pp(v: unknown, digits = 1): string {
  return isNum(v) ? `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(digits)} pp` : "—";
}

export function dateTime(v: unknown): string {
  if (typeof v !== "string") return "—";
  const d = new Date(v);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function shortId(v: unknown, n = 8): string {
  return typeof v === "string" ? v.slice(0, n) : "—";
}
