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
    if (status === "success") return "bg-green-100 text-green-800";
    if (status === "failed") return "bg-red-100 text-red-800";
    if (status === "partial") return "bg-yellow-100 text-yellow-800";
    return "bg-gray-100 text-gray-600";
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading...</div></div>;
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center my-8">
        <p className="text-red-700 font-medium">Failed to load status data</p>
        <p className="text-red-600 text-sm mt-1">{error}</p>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold text-brand-700 mb-6">Data Status</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {sources.map((src) => {
          const lr = lastRun(src);
          return (
            <div key={src} className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
              <p className="text-sm font-semibold text-gray-700">{src.replace(/_/g, " ").toUpperCase()}</p>
              {lr ? (
                <>
                  <span className={`inline-block mt-2 px-2 py-0.5 rounded text-xs font-medium ${statusColor(lr.status)}`}>
                    {lr.status}
                  </span>
                  <p className="text-xs text-gray-400 mt-2">
                    Last run: {new Date(lr.started_at).toLocaleString("en-IN")}
                  </p>
                  <p className="text-xs text-gray-400">
                    Files: {lr.files_downloaded} | Records: {lr.records_written}
                  </p>
                </>
              ) : (
                <p className="text-xs text-gray-400 mt-2">No runs recorded</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Recent Scraper Runs</h3>
        {runs.length === 0 ? (
          <p className="text-gray-400 text-sm">No scraper runs recorded yet. Runs will appear here after the monthly pipeline executes.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-gray-500">
                <th className="pb-2">Source</th>
                <th className="pb-2">Status</th>
                <th className="pb-2">Started</th>
                <th className="pb-2 text-right">Files</th>
                <th className="pb-2 text-right">Records</th>
                <th className="pb-2">Error</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.id} className="border-b last:border-0">
                  <td className="py-2 font-medium">{r.source}</td>
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${statusColor(r.status)}`}>
                      {r.status}
                    </span>
                  </td>
                  <td className="py-2 text-gray-500">{new Date(r.started_at).toLocaleString("en-IN")}</td>
                  <td className="py-2 text-right">{r.files_downloaded}</td>
                  <td className="py-2 text-right">{r.records_written}</td>
                  <td className="py-2 text-gray-400 text-xs truncate max-w-xs">{r.error_message || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
