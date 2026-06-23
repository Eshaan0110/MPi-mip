import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "MIP — Market Intelligence Platform",
  description: "India credit & debit card market forecasting dashboard",
};

function Nav() {
  const links = [
    { href: "/", label: "Dashboard" },
    { href: "/banks", label: "Bank Explorer" },
    { href: "/status", label: "Data Status" },
    { href: "/models", label: "Model Performance" },
    { href: "/about", label: "About" },
  ];

  return (
    <nav className="bg-brand-700 text-white">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center gap-8">
        <span className="font-bold text-lg tracking-wide">MIP</span>
        <div className="flex gap-6 text-sm">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="hover:text-brand-100 transition-colors"
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="max-w-7xl mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
