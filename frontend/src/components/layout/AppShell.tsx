import { NavLink, Outlet } from "react-router-dom";
import { Database, GraduationCap, Search } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Data", icon: Database, step: "1" },
  { to: "/learn", label: "Learn", icon: GraduationCap, step: "2" },
  { to: "/extract", label: "Extract", icon: Search, step: "3" },
];

export function AppShell() {
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-14 w-full max-w-screen-xl items-center gap-6 px-4 sm:px-6">
          <div className="flex items-center gap-2.5 shrink-0">
            <img
              src="/mascot.png"
              alt="RuleChef mascot"
              className="h-8 w-8 rounded-md border border-slate-200 bg-white object-cover p-0.5"
            />
            <h1 className="text-base font-semibold tracking-tight">
              RuleChef <span className="brand-accent">App</span>
            </h1>
          </div>

          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map(({ to, label, icon: Icon, step }) => (
              <NavLink
                key={to}
                to={to}
                end={to === "/"}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
                  )
                }
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{step}. {label}</span>
              </NavLink>
            ))}
          </nav>

          <p className="ml-auto hidden text-xs text-slate-400 md:block">
            Deterministic rule learning workspace
          </p>
        </div>
      </header>

      <main className="mx-auto w-full max-w-screen-xl px-4 py-6 sm:px-6">
        <Outlet />
      </main>
    </div>
  );
}
