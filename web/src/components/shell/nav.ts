import {
  Activity,
  BookOpen,
  Boxes,
  Calculator,
  Gauge,
  LayoutDashboard,
  ListChecks,
  PieChart,
  Scale,
  SlidersHorizontal,
  type LucideIcon,
} from "lucide-react";

export type NavItem = { href: string; label: string; icon: LucideIcon; keywords: string };

export const NAV: NavItem[] = [
  { href: "/", label: "Overview", icon: LayoutDashboard, keywords: "home kpi dashboard" },
  { href: "/score", label: "Score an applicant", icon: Gauge, keywords: "pd decide what-if predict" },
  { href: "/decisions", label: "Decisions log", icon: ListChecks, keywords: "audit applicants history" },
  { href: "/portfolio", label: "Portfolio", icon: PieChart, keywords: "exposure expected loss bands segments hhi" },
  { href: "/policies", label: "Policy simulator", icon: SlidersHorizontal, keywords: "rules thresholds yaml cost matrix" },
  { href: "/scenarios", label: "Scenarios", icon: Calculator, keywords: "economics stress npv break-even" },
  { href: "/models", label: "Models", icon: Boxes, keywords: "registry versions promote roc calibration" },
  { href: "/fairness", label: "Fairness", icon: Scale, keywords: "groups disparity proxy" },
  { href: "/monitoring", label: "Monitoring", icon: Activity, keywords: "drift psi alerts" },
  { href: "/docs", label: "Docs & about", icon: BookOpen, keywords: "architecture methodology limitations" },
];
