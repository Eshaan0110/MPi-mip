import type { Metadata } from "next";
import "./globals.css";
import { NavBar } from "@/components/NavBar";
import { ThemeProvider } from "@/components/ThemeProvider";

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
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col bg-gray-50 text-gray-800 dark:bg-[#0f172a] dark:text-slate-200 transition-colors duration-200">
        <ThemeProvider>
          <NavBar />
          <main className="max-w-7xl mx-auto px-4 py-6 flex-1 w-full">{children}</main>
          <footer className="border-t border-gray-200 dark:border-slate-700/50 mt-8">
            <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between text-xs text-gray-500 dark:text-slate-500">
              <span>MPi Market Intelligence Platform</span>
              <span>Data sourced from RBI &amp; NPCI</span>
            </div>
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
