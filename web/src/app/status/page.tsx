"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";
import type { ScraperRun } from "@/lib/types";

export default function DataStatusPage() {
  const [runs, setRuns] = useState<ScraperRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const { data, error: err } = await supabase
        .from("scraper_runs")
        .select("*")
        .order("started_at", { ascending: false })
        .limit(50);
      if (err) { setError(err.message); setLoading(false); return; }
      if (data) setRuns(data);
      setLoading(false);
    }
    load();
  }, []);

  const sources = ["rbi_bankwise", "rbi_psi", "npci_upi", "rbi_repo"];

  function lastRun(source: string) {
    return runs.find((r) => r.source === source);
  }

  function statusColor(status: string) {
    if (status === "success") return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400";
    if (status === "failed") return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400";
    if (status === "partial") return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400";
    return "bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-400";
  }

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading...</div></div>;

  if (error) return (
    <div className="bg-red-100 border border-red-300 dark:bg-red-900/30 dark:border-red-700/50 rounded-lg p-6 text-center my-8">
      <p className="text-red-700 dark:text-red-400 font-medium">Failed to load status data</p>
      <p className="text-red-500 text-sm mt-1">{error}</p>
    </div>
  );

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">Data Status</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {sources.map((src) => {
          const lr = lastRun(src);
          return (
            <div key={src} className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5">
              <p className="text-sm font-semibold text-gray-800 dark:text-slate-200">{src.replace(/_/g, " ").toUpperCase()}</p>
              {lr ? (
                <>
                  <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-medium ${statusColor(lr.status)}`}>
                    {lr.status}
                  </span>
                  <p className="text-xs text-gray-400 dark:text-slate-500 mt-2">Last run: {new Date(lr.started_at).toLocaleString("en-IN")}</p>
                  <p className="text-xs text-gray-400 dark:text-slate-500">Files: {lr.files_downloaded} | Records: {lr.records_written}</p>
                </>
              ) : (
                <p className="text-xs text-gray-400 dark:text-slate-500 mt-2">No runs recorded</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-4">Recent Scraper Runs</h3>
        {runs.length === 0 ? (
          <p className="text-gray-400 dark:text-slate-500 text-sm">No scraper runs recorded yet. Runs will appear here after the monthly pipeline executes.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                  <th className="pb-3 font-medium">Source</th>
                  <th className="pb-3 font-medium">Status</th>
                  <th className="pb-3 font-medium">Started</th>
                  <th className="pb-3 text-right font-medium">Files</th>
                  <th className="pb-3 text-right font-medium">Records</th>
                  <th className="pb-3 font-medium">Error</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                    <td className="py-3 font-medium text-gray-800 dark:text-slate-200">{r.source}</td>
                    <td className="py-3"><span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColor(r.status)}`}>{r.status}</span></td>
                    <td className="py-3 text-gray-500 dark:text-slate-400">{new Date(r.started_at).toLocaleString("en-IN")}</td>
                    <td className="py-3 text-right text-gray-500 dark:text-slate-400">{r.files_downloaded}</td>
                    <td className="py-3 text-right text-gray-500 dark:text-slate-400">{r.records_written}</td>
                    <td className="py-3 text-gray-400 dark:text-slate-500 text-xs truncate max-w-xs">{r.error_message || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
