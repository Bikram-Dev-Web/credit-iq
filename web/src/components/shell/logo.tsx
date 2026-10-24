/** CreditIQ mark: a rising score line through a rounded square, in the accent colour. */
export function LogoMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" fill="none">
      <rect x="1" y="1" width="30" height="30" rx="8" fill="var(--accent)" />
      <path d="M8 21.5 13.5 16l4 3.5L24 11" stroke="var(--accent-fg)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="24" cy="11" r="2.2" fill="var(--accent-fg)" />
    </svg>
  );
}

export function Wordmark({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <span className="flex items-center gap-2">
      <LogoMark />
      {collapsed ? null : (
        <span className="text-[15px] font-semibold tracking-tight text-fg">
          Credit<span className="text-accent">IQ</span>
        </span>
      )}
    </span>
  );
}
