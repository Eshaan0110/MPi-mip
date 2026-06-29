"use client";

import { useEffect, useState } from "react";
import { supabase } from "@/lib/supabase";

interface Finding {
  id: string;
  signal_type: string;
  bank_name: string | null;
  card_type: string | null;
  title: string;
  impact_direction: string | null;
  impact_magnitude: string | null;
  details: string | null;
  confidence: number;
  source_name: string | null;
  source_url: string | null;
  discovered_at: string;
}

interface RetrainLog {
  id: string;
  metric: string;
  bank_name: string | null;
  old_cv_mape: number | null;
  new_cv_mape: number | null;
  promoted: boolean;
  regressors_used: string[];
  evaluated_at: string;
}

interface AgentRun {
  id: string;
  run_type: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  summary: string;
}

const SIGNAL_LABELS: Record<string, string> = {
  new_card_launch: "New Card Launch",
  card_discontinuation: "Card Discontinued",
  regulatory_change: "Regulatory Change",
  partnership: "Partnership",
  growth_target: "Growth Target",
  macro_policy: "Macro Policy",
  infrastructure_change: "Infrastructure",
  market_event: "Market Event",
};

const SIGNAL_COLORS: Record<string, string> = {
  new_card_launch: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400",
  card_discontinuation: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  regulatory_change: "bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400",
  partnership: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  growth_target: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  macro_policy: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400",
  infrastructure_change: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-400",
  market_event: "bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-300",
};

function directionIcon(dir: string | null): string {
  if (dir === "positive") return "↑";
  if (dir === "negative") return "↓";
  return "↔";
}

function directionColor(dir: string | null): string {
  if (dir === "positive") return "text-emerald-600 dark:text-emerald-400";
  if (dir === "negative") return "text-red-600 dark:text-red-400";
  return "text-gray-500 dark:text-slate-400";
}

function formatDate(d: string): string {
  return new Date(d).toLocaleDateString("en-IN", {
    day: "numeric", month: "short", year: "numeric",
  });
}

export default function AgentPage() {
  const [findings, setFindings] = useState<Finding[]>([]);
  const [retrains, setRetrains] = useState<RetrainLog[]>([]);
  const [runs, setRuns] = useState<AgentRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"findings" | "retrains" | "runs">("findings");

  useEffect(() => {
    async function load() {
      const [f, r, a] = await Promise.all([
        supabase.from("agent_findings").select("*").order("discovered_at", { ascending: false }).limit(100),
        supabase.from("agent_retrains").select("*").order("evaluated_at", { ascending: false }).limit(50),
        supabase.from("agent_runs").select("*").order("started_at", { ascending: false }).limit(20),
      ]);
      if (f.data) setFindings(f.data);
      if (r.data) setRetrains(r.data);
      if (a.data) setRuns(a.data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading agent data...</div></div>;

  const signalCounts: Record<string, number> = {};
  for (const f of findings) {
    signalCounts[f.signal_type] = (signalCounts[f.signal_type] || 0) + 1;
  }
  const promotedCount = retrains.filter((r) => r.promoted).length;
  const lastRun = runs[0];

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">AI Research Agent</h1>
          <p className="text-sm text-gray-400 dark:text-slate-500 mt-0.5">Autonomous market intelligence pipeline</p>
        </div>
        <div className="text-right">
          {lastRun && (
            <div className="text-xs text-gray-400 dark:text-slate-500">
              Last run: {formatDate(lastRun.started_at)} —{" "}
              <span className={lastRun.status === "success" ? "text-emerald-600 dark:text-emerald-400" : lastRun.status === "running" ? "text-amber-600 dark:text-amber-400" : "text-red-600 dark:text-red-400"}>
                {lastRun.status}
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400">Signals Found</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{findings.length}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400">Retrain Attempts</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{retrains.length}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400">Models Improved</p>
          <p className="text-3xl font-bold text-emerald-400">{promotedCount}</p>
        </div>
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5 text-center">
          <p className="text-sm text-gray-500 dark:text-slate-400">Agent Runs</p>
          <p className="text-3xl font-bold text-gray-900 dark:text-white">{runs.length}</p>
        </div>
      </div>

      <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden mb-6 w-fit">
        {(["findings", "retrains", "runs"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${tab === t ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"}`}>
            {t}
          </button>
        ))}
      </div>

      {tab === "findings" && (
        <div className="space-y-3">
          {findings.length === 0 ? (
            <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-12 text-center">
              <p className="text-gray-400 dark:text-slate-500">No findings yet. Run the agent pipeline to start collecting market intelligence.</p>
              <p className="text-xs text-gray-400 dark:text-slate-600 mt-2 font-mono">uv run python -m src.agent</p>
            </div>
          ) : (
            findings.map((f) => (
              <div key={f.id} className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-5">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${SIGNAL_COLORS[f.signal_type] || "bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-300"}`}>
                        {SIGNAL_LABELS[f.signal_type] || f.signal_type}
                      </span>
                      {f.bank_name && (
                        <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-300">{f.bank_name}</span>
                      )}
                      {f.card_type && (
                        <span className="text-xs text-gray-400 dark:text-slate-500">{f.card_type === "CC" ? "Credit" : f.card_type === "DC" ? "Debit" : "Both"}</span>
                      )}
                      <span className={`text-lg ${directionColor(f.impact_direction)}`}>{directionIcon(f.impact_direction)}</span>
                    </div>
                    <p className="text-sm font-medium text-gray-800 dark:text-slate-200">{f.title}</p>
                    {f.details && <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">{f.details}</p>}
                  </div>
                  <div className="text-right shrink-0">
                    <p className="text-xs text-gray-400 dark:text-slate-500">{formatDate(f.discovered_at)}</p>
                    <p className="text-xs text-gray-400 dark:text-slate-600 mt-0.5">{(f.confidence * 100).toFixed(0)}% conf</p>
                  </div>
                </div>
                {f.source_url && (
                  <p className="text-xs text-gray-400 dark:text-slate-600 mt-2 truncate">Source: {f.source_name || f.source_url}</p>
                )}
              </div>
            ))
          )}
        </div>
      )}

      {tab === "retrains" && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
          {retrains.length === 0 ? (
            <p className="text-gray-400 dark:text-slate-500 text-center py-8">No retrain attempts yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                    <th className="pb-3 font-medium">Metric</th>
                    <th className="pb-3 font-medium">Bank</th>
                    <th className="pb-3 text-right font-medium">Old Accuracy</th>
                    <th className="pb-3 text-right font-medium">New Accuracy</th>
                    <th className="pb-3 text-center font-medium">Result</th>
                    <th className="pb-3 font-medium">Regressors</th>
                    <th className="pb-3 text-right font-medium">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {retrains.map((r) => (
                    <tr key={r.id} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                      <td className="py-3 font-medium text-gray-800 dark:text-slate-200">{r.metric?.replace(/_/g, " ")}</td>
                      <td className="py-3 text-gray-500 dark:text-slate-400">{r.bank_name || "Aggregate"}</td>
                      <td className="py-3 text-right text-gray-500 dark:text-slate-400">{r.old_cv_mape != null ? (100 - r.old_cv_mape).toFixed(1) + "%" : "—"}</td>
                      <td className="py-3 text-right text-gray-500 dark:text-slate-400">{r.new_cv_mape != null ? (100 - r.new_cv_mape).toFixed(1) + "%" : "—"}</td>
                      <td className="py-3 text-center">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${r.promoted ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400" : "bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-400"}`}>
                          {r.promoted ? "Promoted" : "Kept Old"}
                        </span>
                      </td>
                      <td className="py-3 text-gray-400 dark:text-slate-500 text-xs">{r.regressors_used?.join(", ") || "—"}</td>
                      <td className="py-3 text-right text-gray-400 dark:text-slate-500">{formatDate(r.evaluated_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {tab === "runs" && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
          {runs.length === 0 ? (
            <p className="text-gray-400 dark:text-slate-500 text-center py-8">No agent runs recorded yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                    <th className="pb-3 font-medium">Type</th>
                    <th className="pb-3 font-medium">Status</th>
                    <th className="pb-3 font-medium">Started</th>
                    <th className="pb-3 font-medium">Duration</th>
                    <th className="pb-3 font-medium">Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => {
                    const duration = r.finished_at
                      ? Math.round((new Date(r.finished_at).getTime() - new Date(r.started_at).getTime()) / 1000)
                      : null;
                    return (
                      <tr key={r.id} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                        <td className="py-3 font-medium text-gray-800 dark:text-slate-200 capitalize">{r.run_type}</td>
                        <td className="py-3">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            r.status === "success" ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
                            : r.status === "running" ? "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
                            : "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                          }`}>{r.status}</span>
                        </td>
                        <td className="py-3 text-gray-500 dark:text-slate-400">{formatDate(r.started_at)}</td>
                        <td className="py-3 text-gray-400 dark:text-slate-500">{duration != null ? `${duration}s` : "running..."}</td>
                        <td className="py-3 text-gray-400 dark:text-slate-500 text-xs truncate max-w-xs">{r.summary || "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
