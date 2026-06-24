"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/banks", label: "Bank Explorer" },
  { href: "/status", label: "Data Status" },
  { href: "/models", label: "Model Performance" },
  { href: "/about", label: "About" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="bg-brand-700 text-white">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
        <span className="font-bold text-lg tracking-wide">MIP</span>
        <div className="flex gap-6 text-sm">
          {links.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`transition-colors ${active ? "text-white font-semibold border-b-2 border-white pb-0.5" : "text-brand-100 hover:text-white"}`}
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
