"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/banks", label: "Bank Explorer" },
  { href: "/builder", label: "Builder" },
  { href: "/status", label: "Data Status" },
  { href: "/models", label: "Model Performance" },
  { href: "/about", label: "About" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-slate-900 border-b border-slate-700/50">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
        <span className="font-bold text-lg tracking-wide text-white">MIP</span>
        <div className="flex gap-6 text-sm overflow-x-auto">
          {links.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`transition-colors whitespace-nowrap ${
                  active
                    ? "text-blue-400 font-semibold border-b-2 border-blue-400 pb-0.5"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
