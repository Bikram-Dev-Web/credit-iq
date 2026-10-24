"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { Command } from "cmdk";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Laptop, Menu, Moon, PanelLeftClose, PanelLeftOpen, Search, ShieldCheck, Sun, UserRound } from "lucide-react";
import { cn } from "@/lib/utils";
import { useMeta, useSession, useSetRole } from "@/lib/api/hooks";
import { Badge, Button, Kbd, Tooltip } from "@/components/ui/primitives";
import { NAV } from "./nav";
import { Wordmark } from "./logo";

const DISCLAIMER = "Educational decision-support prototype, not for real credit decisions.";

/* The nav-collapsed preference lives in localStorage (a per-viewer convenience), read as an external store. */
const NAV_KEY = "ciq-nav-collapsed";
const navListeners = new Set<() => void>();
function readCollapsed(): boolean {
  try {
    return localStorage.getItem(NAV_KEY) === "1";
  } catch {
    return false; // storage unavailable: keep the default
  }
}
function useCollapsed(): [boolean, (v: boolean) => void] {
  const collapsed = React.useSyncExternalStore(
    (cb) => {
      navListeners.add(cb);
      return () => navListeners.delete(cb);
    },
    readCollapsed,
    () => false,
  );
  const set = (v: boolean) => {
    try {
      localStorage.setItem(NAV_KEY, v ? "1" : "0");
    } catch {
      /* ignore */
    }
    navListeners.forEach((cb) => cb());
  };
  return [collapsed, set];
}

const noopSubscribe = () => () => {};
/** False during server render and hydration, true afterwards (no effect + setState round trip). */
function useHydrated(): boolean {
  return React.useSyncExternalStore(noopSubscribe, () => true, () => false);
}

function NavLinks({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav aria-label="Main" className="flex flex-col gap-0.5">
      {NAV.map((item) => {
        const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        const link = (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors duration-150",
              active ? "bg-surface-2 font-medium text-fg" : "text-fg-secondary hover:bg-surface-2 hover:text-fg",
              collapsed && "justify-center px-0",
            )}
          >
            <item.icon aria-hidden className={cn("size-4 shrink-0", active ? "text-accent" : "text-fg-muted")} />
            {collapsed ? <span className="sr-only">{item.label}</span> : <span className="truncate">{item.label}</span>}
          </Link>
        );
        return collapsed ? (
          <Tooltip key={item.href} content={item.label} side="right">
            {link}
          </Tooltip>
        ) : (
          link
        );
      })}
    </nav>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const mounted = useHydrated();
  const next = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
  const Icon = !mounted ? Laptop : theme === "dark" ? Moon : theme === "light" ? Sun : Laptop;
  return (
    <Tooltip content={`Theme: ${mounted ? theme : "system"} (click for ${next})`}>
      <Button variant="ghost" size="icon" aria-label={`Switch theme to ${next}`} onClick={() => setTheme(next)}>
        <Icon />
      </Button>
    </Tooltip>
  );
}

function RoleSwitch() {
  const { data } = useSession();
  const setRole = useSetRole();
  if (!data) return null;
  if (!data.adminAvailable) {
    return (
      <Badge tone="neutral" title="Admin actions are disabled on this deployment">
        <UserRound className="size-3" /> analyst
      </Badge>
    );
  }
  const admin = data.role === "admin";
  return (
    <Tooltip content={admin ? "Admin: can promote models and save policies. Click for analyst." : "Analyst: read and score. Click for admin."}>
      <Button
        variant={admin ? "outline" : "ghost"}
        size="sm"
        onClick={() => setRole.mutate(admin ? "analyst" : "admin")}
        aria-label={`Role: ${data.role}. Switch to ${admin ? "analyst" : "admin"}`}
        data-testid="role-switch"
      >
        {admin ? <ShieldCheck className="text-accent" /> : <UserRound />}
        <span className="hidden sm:inline">{data.role}</span>
      </Button>
    </Tooltip>
  );
}

function VersionPills() {
  const { data: meta, isError } = useMeta();
  if (isError) return <Badge tone="critical">API unreachable</Badge>;
  if (!meta) return <div className="skeleton h-5 w-40" />;
  return (
    <div className="hidden items-center gap-1.5 md:flex">
      <Tooltip content={`${meta.model_name} — ${meta.model_label ?? "no model"}`}>
        <Link href="/models">
          <Badge tone="accent" className="font-mono">
            model v{meta.model_version ?? "—"}
          </Badge>
        </Link>
      </Tooltip>
      <Tooltip content={meta.policy_name ?? "No active policy"}>
        <Link href="/policies">
          <Badge className="font-mono">policy {meta.policy_version ?? "—"}</Badge>
        </Link>
      </Tooltip>
      {meta.data_provenance === "synthetic" ? (
        <Tooltip content="The served model was trained on fabricated demonstration data.">
          <Badge tone="serious">synthetic data</Badge>
        </Tooltip>
      ) : null}
    </div>
  );
}

function EnvBadge() {
  const { data: meta } = useMeta();
  const env = meta?.environment ?? process.env.NEXT_PUBLIC_ENVIRONMENT ?? "local";
  return (
    <Badge tone={env === "production" ? "critical" : env === "demo" ? "warning" : "neutral"} className="uppercase tracking-wide">
      {env}
    </Badge>
  );
}

function CommandPalette({ open, setOpen }: { open: boolean; setOpen: (o: boolean) => void }) {
  const router = useRouter();
  const { setTheme } = useTheme();
  const setRole = useSetRole();
  const { data: session } = useSession();
  const run = (fn: () => void) => {
    setOpen(false);
    fn();
  };
  return (
    <DialogPrimitive.Root open={open} onOpenChange={setOpen}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" />
        <DialogPrimitive.Content className="fixed left-1/2 top-[15vh] z-50 w-[calc(100vw-2rem)] max-w-lg -translate-x-1/2 overflow-hidden rounded-lg border border-border bg-surface shadow-card">
          <DialogPrimitive.Title className="sr-only">Command palette</DialogPrimitive.Title>
          <DialogPrimitive.Description className="sr-only">Jump to a page or run an action</DialogPrimitive.Description>
          <Command label="Command palette" className="flex flex-col">
            <div className="flex items-center gap-2 border-b border-border px-3">
              <Search aria-hidden className="size-4 text-fg-muted" />
              <Command.Input placeholder="Jump to a page or run an action…" className="h-11 w-full bg-transparent text-sm text-fg outline-none placeholder:text-fg-muted" />
            </div>
            <Command.List className="max-h-80 overflow-y-auto p-1.5">
              <Command.Empty className="px-3 py-6 text-center text-xs text-fg-muted">No matches.</Command.Empty>
              <Command.Group heading="Pages" className="px-1 text-[11px] text-fg-muted [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                {NAV.map((n) => (
                  <Command.Item
                    key={n.href}
                    value={`${n.label} ${n.keywords}`}
                    onSelect={() => run(() => router.push(n.href))}
                    className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-2 text-[13px] text-fg data-[selected=true]:bg-surface-2"
                  >
                    <n.icon aria-hidden className="size-4 text-fg-muted" />
                    {n.label}
                  </Command.Item>
                ))}
              </Command.Group>
              <Command.Group heading="Actions" className="px-1 text-[11px] text-fg-muted [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5">
                {(["dark", "light", "system"] as const).map((t) => (
                  <Command.Item
                    key={t}
                    value={`theme ${t}`}
                    onSelect={() => run(() => setTheme(t))}
                    className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-2 text-[13px] text-fg data-[selected=true]:bg-surface-2"
                  >
                    {t === "dark" ? <Moon className="size-4 text-fg-muted" /> : t === "light" ? <Sun className="size-4 text-fg-muted" /> : <Laptop className="size-4 text-fg-muted" />}
                    Theme: {t}
                  </Command.Item>
                ))}
                {session?.adminAvailable ? (
                  <Command.Item
                    value="switch role admin analyst"
                    onSelect={() => run(() => setRole.mutate(session.role === "admin" ? "analyst" : "admin"))}
                    className="flex cursor-pointer items-center gap-2.5 rounded-md px-2 py-2 text-[13px] text-fg data-[selected=true]:bg-surface-2"
                  >
                    <ShieldCheck className="size-4 text-fg-muted" />
                    Switch role to {session.role === "admin" ? "analyst" : "admin"}
                  </Command.Item>
                ) : null}
              </Command.Group>
            </Command.List>
          </Command>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useCollapsed();
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [paletteOpen, setPaletteOpen] = React.useState(false);

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="flex min-h-dvh">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-accent focus:px-3 focus:py-1.5 focus:text-accent-fg">
        Skip to content
      </a>
      {/* desktop sidebar */}
      <aside
        className={cn(
          "sticky top-0 hidden h-dvh shrink-0 flex-col border-r border-border bg-bg transition-[width] duration-200 lg:flex",
          collapsed ? "w-14" : "w-56",
        )}
      >
        <div className={cn("flex h-14 items-center border-b border-border", collapsed ? "justify-center" : "px-4")}>
          <Link href="/" aria-label="CreditIQ home">
            <Wordmark collapsed={collapsed} />
          </Link>
        </div>
        <div className={cn("flex-1 overflow-y-auto py-3", collapsed ? "px-2" : "px-2.5")}>
          <NavLinks collapsed={collapsed} />
        </div>
        <div className={cn("border-t border-border p-2", collapsed && "flex justify-center")}>
          <Button
            variant="ghost"
            size={collapsed ? "icon" : "sm"}
            className={collapsed ? "" : "w-full justify-start"}
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
          >
            {collapsed ? <PanelLeftOpen /> : <PanelLeftClose />}
            {collapsed ? null : "Collapse"}
          </Button>
        </div>
      </aside>

      {/* mobile drawer */}
      <DialogPrimitive.Root open={mobileOpen} onOpenChange={setMobileOpen}>
        <DialogPrimitive.Portal>
          <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/50 lg:hidden" />
          <DialogPrimitive.Content className="fixed inset-y-0 left-0 z-50 w-64 border-r border-border bg-bg p-3 lg:hidden">
            <DialogPrimitive.Title className="mb-4 px-1">
              <Wordmark />
            </DialogPrimitive.Title>
            <DialogPrimitive.Description className="sr-only">Navigation</DialogPrimitive.Description>
            <NavLinks collapsed={false} onNavigate={() => setMobileOpen(false)} />
          </DialogPrimitive.Content>
        </DialogPrimitive.Portal>
      </DialogPrimitive.Root>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b border-border bg-bg/85 px-4 backdrop-blur supports-[backdrop-filter]:bg-bg/70">
          <Button variant="ghost" size="icon" className="lg:hidden" aria-label="Open navigation" onClick={() => setMobileOpen(true)}>
            <Menu />
          </Button>
          <div className="lg:hidden">
            <Wordmark collapsed />
          </div>
          <VersionPills />
          <div className="ml-auto flex items-center gap-1.5">
            <EnvBadge />
            <Button
              variant="outline"
              size="sm"
              onClick={() => setPaletteOpen(true)}
              aria-label="Open command palette"
              className="hidden text-fg-muted sm:inline-flex"
            >
              <Search /> Search <Kbd>⌘K</Kbd>
            </Button>
            <Button variant="ghost" size="icon" className="sm:hidden" aria-label="Open command palette" onClick={() => setPaletteOpen(true)}>
              <Search />
            </Button>
            <RoleSwitch />
            <ThemeToggle />
          </div>
        </header>
        <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-6 sm:px-6">
          {children}
        </main>
        <footer className="border-t border-border px-4 py-3 text-center text-[11px] text-fg-muted sm:px-6">
          {DISCLAIMER} Outcomes are “within threshold” or “flagged for review”, never approval or denial.
        </footer>
      </div>
      <CommandPalette open={paletteOpen} setOpen={setPaletteOpen} />
    </div>
  );
}
