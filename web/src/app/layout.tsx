import type { Metadata } from "next";
import "./globals.css";
import { NavBar } from "@/components/NavBar";

export const metadata: Metadata = {
  title: "MIP — Market Intelligence Platform",
  description: "India credit & debit card market forecasting dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-[#0f172a] text-slate-200">
        <NavBar />
        <main className="max-w-7xl mx-auto px-4 py-6 flex-1 w-full">{children}</main>
        <footer className="border-t border-slate-700/50 mt-8">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between text-xs text-slate-500">
            <span>MPi Market Intelligence Platform</span>
            <span>Data sourced from RBI &amp; NPCI</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
