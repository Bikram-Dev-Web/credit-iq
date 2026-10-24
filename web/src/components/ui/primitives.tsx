"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import * as SwitchPrimitive from "@radix-ui/react-switch";
import * as SliderPrimitive from "@radix-ui/react-slider";
import * as LabelPrimitive from "@radix-ui/react-label";
import * as SelectPrimitive from "@radix-ui/react-select";
import { cva, type VariantProps } from "class-variance-authority";
import { Check, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ Button */
const buttonVariants = cva(
  "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-md text-[13px] font-medium transition-colors duration-150 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-accent text-accent-fg hover:bg-accent-hover",
        secondary: "border border-border bg-surface-2 text-fg hover:bg-surface-3",
        ghost: "text-fg-secondary hover:bg-surface-2 hover:text-fg",
        outline: "border border-border text-fg hover:bg-surface-2",
        danger: "bg-critical text-white hover:opacity-90",
      },
      size: { sm: "h-7 px-2.5", md: "h-8 px-3", lg: "h-9 px-4 text-sm", icon: "size-8" },
    },
    defaultVariants: { variant: "secondary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return <Comp ref={ref} className={cn(buttonVariants({ variant, size }), className)} {...props} />;
  },
);
Button.displayName = "Button";

/* ------------------------------------------------------------------ Card */
export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...props} />;
}

export function CardHeader({
  title,
  description,
  actions,
  className,
}: {
  title: React.ReactNode;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-3", className)}>
      <div className="min-w-0">
        <h3 className="text-[13px] font-semibold text-fg">{title}</h3>
        {description ? <p className="mt-0.5 text-xs text-fg-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-4", className)} {...props} />;
}

/* ------------------------------------------------------------------ Badge */
const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-4 whitespace-nowrap",
  {
    variants: {
      tone: {
        neutral: "border-border bg-surface-2 text-fg-secondary",
        accent: "border-transparent bg-accent-soft text-accent",
        good: "border-transparent bg-good-soft text-fg",
        warning: "border-transparent bg-warning-soft text-fg",
        serious: "border-transparent bg-serious-soft text-fg",
        critical: "border-transparent bg-critical-soft text-fg",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export function Badge({
  className,
  tone,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}

/* ------------------------------------------------------------------ Inputs */
export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-8 w-full rounded-md border border-border bg-surface px-2.5 text-[13px] text-fg placeholder:text-fg-muted",
        "transition-colors duration-150 hover:border-border-strong focus-visible:border-accent focus-visible:outline-none",
        "aria-[invalid=true]:border-critical disabled:opacity-50",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded-md border border-border bg-surface px-3 py-2 font-mono text-xs text-fg",
        "focus-visible:border-accent focus-visible:outline-none",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

export function Label({ className, ...props }: React.ComponentProps<typeof LabelPrimitive.Root>) {
  return <LabelPrimitive.Root className={cn("text-xs font-medium text-fg-secondary", className)} {...props} />;
}

export function Field({
  label,
  hint,
  error,
  htmlFor,
  children,
  className,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  error?: string;
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-w-0 flex-col gap-1", className)}>
      <Label htmlFor={htmlFor} className="flex min-h-[18px] flex-wrap items-center gap-1">
        {label}
      </Label>
      {children}
      {error ? (
        <p className="text-[11px] text-critical" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="truncate text-[11px] text-fg-muted">{hint}</p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ Select */
export function Select({
  value,
  onValueChange,
  options,
  placeholder,
  id,
  ariaLabel,
  className,
}: {
  value: string | undefined;
  onValueChange: (v: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  id?: string;
  ariaLabel?: string;
  className?: string;
}) {
  return (
    <SelectPrimitive.Root value={value} onValueChange={onValueChange}>
      <SelectPrimitive.Trigger
        id={id}
        aria-label={ariaLabel}
        className={cn(
          "inline-flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-md border border-border bg-surface px-2.5 text-[13px] text-fg",
          "hover:border-border-strong focus-visible:border-accent focus-visible:outline-none",
          className,
        )}
      >
        <span className="truncate">
          <SelectPrimitive.Value placeholder={placeholder} />
        </span>
        <SelectPrimitive.Icon>
          <ChevronDown className="size-3.5 text-fg-muted" />
        </SelectPrimitive.Icon>
      </SelectPrimitive.Trigger>
      <SelectPrimitive.Portal>
        <SelectPrimitive.Content
          position="popper"
          sideOffset={4}
          className="z-50 max-h-72 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-md border border-border bg-surface-2 shadow-card"
        >
          <SelectPrimitive.Viewport className="p-1">
            {options.map((o) => (
              <SelectPrimitive.Item
                key={o.value}
                value={o.value}
                className="relative flex cursor-pointer select-none items-center rounded-sm py-1.5 pl-7 pr-2 text-[13px] text-fg outline-none data-[highlighted]:bg-surface-3"
              >
                <SelectPrimitive.ItemIndicator className="absolute left-2">
                  <Check className="size-3.5 text-accent" />
                </SelectPrimitive.ItemIndicator>
                <SelectPrimitive.ItemText>{o.label}</SelectPrimitive.ItemText>
              </SelectPrimitive.Item>
            ))}
          </SelectPrimitive.Viewport>
        </SelectPrimitive.Content>
      </SelectPrimitive.Portal>
    </SelectPrimitive.Root>
  );
}

/* ------------------------------------------------------------------ Switch, Slider */
export function Switch({ className, ...props }: React.ComponentProps<typeof SwitchPrimitive.Root>) {
  return (
    <SwitchPrimitive.Root
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border border-border bg-surface-3 transition-colors duration-150 data-[state=checked]:bg-accent",
        className,
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb className="block size-4 translate-x-0.5 rounded-full bg-fg shadow transition-transform duration-150 data-[state=checked]:translate-x-[18px] data-[state=checked]:bg-accent-fg" />
    </SwitchPrimitive.Root>
  );
}

export function Slider({ className, ...props }: React.ComponentProps<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root className={cn("relative flex h-5 w-full touch-none select-none items-center", className)} {...props}>
      <SliderPrimitive.Track className="relative h-1 grow overflow-hidden rounded-full bg-surface-3">
        <SliderPrimitive.Range className="absolute h-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        aria-label={props["aria-label"]}
        className="block size-3.5 rounded-full border-2 border-accent bg-surface shadow focus-visible:outline-2 focus-visible:outline-ring"
      />
    </SliderPrimitive.Root>
  );
}

/* ------------------------------------------------------------------ Tabs */
export const Tabs = TabsPrimitive.Root;

export function TabsList({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.List>) {
  return (
    <TabsPrimitive.List
      className={cn("inline-flex h-8 items-center gap-0.5 rounded-md border border-border bg-surface-2 p-0.5", className)}
      {...props}
    />
  );
}

export function TabsTrigger({ className, ...props }: React.ComponentProps<typeof TabsPrimitive.Trigger>) {
  return (
    <TabsPrimitive.Trigger
      className={cn(
        "inline-flex h-full items-center rounded-[5px] px-2.5 text-xs font-medium text-fg-muted transition-colors duration-150",
        "hover:text-fg data-[state=active]:bg-surface data-[state=active]:text-fg data-[state=active]:shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

export const TabsContent = TabsPrimitive.Content;

/* ------------------------------------------------------------------ Tooltip */
export const TooltipProvider = TooltipPrimitive.Provider;

export function Tooltip({ content, children, side = "top" }: { content: React.ReactNode; children: React.ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <TooltipPrimitive.Root delayDuration={150}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-72 rounded-md border border-border bg-surface-2 px-2.5 py-1.5 text-xs text-fg shadow-card"
        >
          {content}
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}

/* ------------------------------------------------------------------ Skeleton, empty, error, separators */
export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("skeleton h-4 w-full", className)} />;
}

export function EmptyState({
  icon,
  title,
  body,
  action,
  className,
}: {
  icon?: React.ReactNode;
  title: string;
  body: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-6 py-10 text-center", className)}>
      {icon ? <div className="text-fg-muted [&_svg]:size-6">{icon}</div> : null}
      <p className="text-sm font-medium text-fg">{title}</p>
      <p className="max-w-md text-xs text-fg-muted">{body}</p>
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ error, className }: { error: unknown; className?: string }) {
  const e = error as { message?: string; errors?: string[] };
  return (
    <div role="alert" className={cn("rounded-md border border-critical/40 bg-critical-soft px-3 py-2 text-xs text-fg", className)}>
      <p className="font-medium">{e?.message ?? "Something went wrong."}</p>
      {e?.errors?.length ? (
        <ul className="mt-1 list-disc pl-4 text-fg-secondary">
          {e.errors.slice(0, 8).map((m) => (
            <li key={m} className="font-mono">
              {m}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

export function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="rounded border border-border bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-fg-muted">
      {children}
    </kbd>
  );
}

export function Stat({
  label,
  value,
  sub,
  className,
  provenance,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  sub?: React.ReactNode;
  className?: string;
  provenance?: React.ReactNode;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <div className="whitespace-nowrap text-[11px] font-medium uppercase tracking-wide text-fg-muted">{label}</div>
      {provenance ? <div className="mt-1">{provenance}</div> : null}
      <div className="num mt-1 font-mono text-lg font-semibold text-fg">{value}</div>
      {sub ? <div className="mt-0.5 text-xs text-fg-muted">{sub}</div> : null}
    </div>
  );
}
