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
    if (status === "success") return "bg-emerald-900/30 text-emerald-400";
    if (status === "failed") return "bg-red-900/30 text-red-400";
    if (status === "partial") return "bg-amber-900/30 text-amber-400";
    return "bg-slate-700 text-slate-400";
  }

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-slate-500">Loading...</div></div>;

  if (error) return (
    <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-6 text-center my-8">
      <p className="text-red-400 font-medium">Failed to load status data</p>
      <p className="text-red-500 text-sm mt-1">{error}</p>
    </div>
  );

  return (
    <div>
      <h1 className="text-2xl font-bold text-white mb-6">Data Status</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {sources.map((src) => {
          const lr = lastRun(src);
          return (
            <div key={src} className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-5">
              <p className="text-sm font-semibold text-slate-200">{src.replace(/_/g, " ").toUpperCase()}</p>
              {lr ? (
                <>
                  <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-medium ${statusColor(lr.status)}`}>
                    {lr.status}
                  </span>
                  <p className="text-xs text-slate-500 mt-2">Last run: {new Date(lr.started_at).toLocaleString("en-IN")}</p>
                  <p className="text-xs text-slate-500">Files: {lr.files_downloaded} | Records: {lr.records_written}</p>
                </>
              ) : (
                <p className="text-xs text-slate-500 mt-2">No runs recorded</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6">
        <h3 className="text-sm font-semibold text-slate-200 mb-4">Recent Scraper Runs</h3>
        {runs.length === 0 ? (
          <p className="text-slate-500 text-sm">No scraper runs recorded yet. Runs will appear here after the monthly pipeline executes.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-slate-400">
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
                  <tr key={r.id} className="border-b border-slate-700/50 last:border-0 hover:bg-slate-700/30">
                    <td className="py-3 font-medium text-slate-200">{r.source}</td>
                    <td className="py-3"><span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColor(r.status)}`}>{r.status}</span></td>
                    <td className="py-3 text-slate-400">{new Date(r.started_at).toLocaleString("en-IN")}</td>
                    <td className="py-3 text-right text-slate-400">{r.files_downloaded}</td>
                    <td className="py-3 text-right text-slate-400">{r.records_written}</td>
                    <td className="py-3 text-slate-500 text-xs truncate max-w-xs">{r.error_message || "—"}</td>
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
