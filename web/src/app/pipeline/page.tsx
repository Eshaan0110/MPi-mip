"use client";

import { useState, useEffect, useCallback } from "react";

interface WorkflowRun {
  id: number;
  status: string;
  conclusion: string | null;
  created_at: string;
  updated_at: string;
  html_url: string;
  event: string;
  run_number: number;
}

type PipelineType = "data" | "agent";

function statusBadge(status: string, conclusion: string | null) {
  if (status === "completed") {
    if (conclusion === "success") return <span className="px-2 py-0.5 rounded text-xs font-medium bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400">Success</span>;
    if (conclusion === "failure") return <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400">Failed</span>;
    return <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-400">{conclusion}</span>;
  }
  if (status === "in_progress") return <span className="px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 animate-pulse">Running</span>;
  if (status === "queued") return <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400">Queued</span>;
  return <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-200 text-gray-600 dark:bg-slate-700 dark:text-slate-400">{status}</span>;
}

function RunsTable({ runs, configured, fetchRuns }: { runs: WorkflowRun[]; configured: boolean; fetchRuns: () => void }) {
  return (
    <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200">Recent Runs</h3>
        <button onClick={fetchRuns} className="text-xs px-3 py-1.5 rounded-md bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-600 transition-colors">
          Refresh
        </button>
      </div>
      {runs.length === 0 ? (
        <p className="text-gray-400 dark:text-slate-500 text-sm">{configured ? "No runs found. Trigger one above." : "Configure GitHub integration first."}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-slate-700 text-left text-gray-500 dark:text-slate-400">
                <th className="pb-3 font-medium">#</th>
                <th className="pb-3 font-medium">Status</th>
                <th className="pb-3 font-medium">Trigger</th>
                <th className="pb-3 font-medium">Started</th>
                <th className="pb-3 font-medium">Updated</th>
                <th className="pb-3 font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id} className="border-b border-gray-200 dark:border-slate-700/50 last:border-0 hover:bg-gray-50 dark:hover:bg-slate-700/30">
                  <td className="py-3 font-medium text-gray-800 dark:text-slate-200">#{run.run_number}</td>
                  <td className="py-3">{statusBadge(run.status, run.conclusion)}</td>
                  <td className="py-3 text-gray-500 dark:text-slate-400 capitalize">{run.event.replace("_", " ")}</td>
                  <td className="py-3 text-gray-500 dark:text-slate-400">{new Date(run.created_at).toLocaleString("en-IN")}</td>
                  <td className="py-3 text-gray-500 dark:text-slate-400">{new Date(run.updated_at).toLocaleString("en-IN")}</td>
                  <td className="py-3">
                    <a href={run.html_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline text-xs">
                      View on GitHub
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function PipelinePage() {
  const [tab, setTab] = useState<PipelineType>("data");

  const [dataRuns, setDataRuns] = useState<WorkflowRun[]>([]);
  const [dataConfigured, setDataConfigured] = useState(true);
  const [agentRuns, setAgentRuns] = useState<WorkflowRun[]>([]);
  const [agentConfigured, setAgentConfigured] = useState(true);

  const [triggering, setTriggering] = useState(false);
  const [message, setMessage] = useState<{ text: string; success: boolean } | null>(null);

  const [skipScraping, setSkipScraping] = useState(false);
  const [skipTraining, setSkipTraining] = useState(false);
  const [skipRetrain, setSkipRetrain] = useState(false);

  const [loading, setLoading] = useState(true);

  const fetchDataRuns = useCallback(async () => {
    try {
      const res = await fetch("/api/pipeline");
      const data = await res.json();
      setDataRuns(data.runs || []);
      setDataConfigured(data.configured !== false);
    } catch {}
  }, []);

  const fetchAgentRuns = useCallback(async () => {
    try {
      const res = await fetch("/api/agent-pipeline");
      const data = await res.json();
      setAgentRuns(data.runs || []);
      setAgentConfigured(data.configured !== false);
    } catch {}
  }, []);

  useEffect(() => {
    Promise.all([fetchDataRuns(), fetchAgentRuns()]).finally(() => setLoading(false));
  }, [fetchDataRuns, fetchAgentRuns]);

  const runs = tab === "data" ? dataRuns : agentRuns;
  const configured = tab === "data" ? dataConfigured : agentConfigured;
  const fetchRuns = tab === "data" ? fetchDataRuns : fetchAgentRuns;

  useEffect(() => {
    if (runs.some((r) => r.status === "in_progress" || r.status === "queued")) {
      const interval = setInterval(fetchRuns, 10000);
      return () => clearInterval(interval);
    }
  }, [runs, fetchRuns]);

  const triggerPipeline = async () => {
    setTriggering(true);
    setMessage(null);
    const endpoint = tab === "data" ? "/api/pipeline" : "/api/agent-pipeline";
    const payload = tab === "data"
      ? { skipScraping, skipTraining }
      : { skipRetrain };

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage({ text: "Pipeline triggered! It will appear below shortly.", success: true });
        setTimeout(fetchRuns, 5000);
      } else {
        setMessage({ text: data.error || "Failed to trigger pipeline", success: false });
      }
    } catch (err: unknown) {
      setMessage({ text: err instanceof Error ? err.message : "Network error", success: false });
    } finally {
      setTriggering(false);
    }
  };

  if (loading) return <div className="flex items-center justify-center h-64"><div className="text-gray-400 dark:text-slate-500">Loading...</div></div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Run Pipeline</h1>
          <p className="text-sm text-gray-500 dark:text-slate-400 mt-1">
            Trigger data or agent pipelines via GitHub Actions
          </p>
        </div>
      </div>

      <div className="flex rounded-lg border border-gray-300 dark:border-slate-600 overflow-hidden mb-6 w-fit">
        {(["data", "agent"] as const).map((t) => (
          <button key={t} onClick={() => { setTab(t); setMessage(null); }}
            className={`px-5 py-2 text-sm font-medium transition-colors ${
              tab === t ? "bg-blue-600 text-white" : "bg-white dark:bg-slate-800 text-gray-500 dark:text-slate-400 hover:bg-gray-200 dark:hover:bg-slate-700"
            }`}>
            {t === "data" ? "Data Pipeline" : "Agent Pipeline"}
          </button>
        ))}
      </div>

      {!configured && (
        <div className="bg-amber-50 border border-amber-200 dark:bg-amber-900/20 dark:border-amber-700/50 rounded-lg p-4 mb-6">
          <p className="text-amber-700 dark:text-amber-400 text-sm font-medium">Pipeline not configured</p>
          <p className="text-amber-600 dark:text-amber-500 text-xs mt-1">
            Set <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">GITHUB_REPO</code> and <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">GITHUB_PAT</code> environment variables.
            {tab === "agent" && <> Also add <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">ANTHROPIC_API_KEY</code> as a GitHub secret.</>}
          </p>
        </div>
      )}

      {configured && (
        <div className="bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6 mb-6">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-4">
            Trigger {tab === "data" ? "Data" : "Agent"} Pipeline
          </h3>
          <div className="flex flex-wrap items-end gap-4">
            {tab === "data" ? (
              <>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
                  <input type="checkbox" checked={skipScraping} onChange={(e) => setSkipScraping(e.target.checked)} className="rounded border-gray-300 dark:border-slate-600" />
                  Skip scraping
                </label>
                <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
                  <input type="checkbox" checked={skipTraining} onChange={(e) => setSkipTraining(e.target.checked)} className="rounded border-gray-300 dark:border-slate-600" />
                  Skip training
                </label>
              </>
            ) : (
              <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-slate-400">
                <input type="checkbox" checked={skipRetrain} onChange={(e) => setSkipRetrain(e.target.checked)} className="rounded border-gray-300 dark:border-slate-600" />
                Skip retraining (research + extract only)
              </label>
            )}
            <button onClick={triggerPipeline} disabled={triggering}
              className={`px-6 py-2.5 rounded-lg font-semibold text-sm transition-all ${
                triggering
                  ? "bg-gray-300 dark:bg-slate-700 text-gray-500 dark:text-slate-400 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white shadow-sm hover:shadow"
              }`}>
              {triggering ? "Triggering..." : `Run ${tab === "data" ? "Data" : "Agent"} Pipeline`}
            </button>
          </div>
          {message && (
            <div className={`mt-4 rounded-lg p-3 text-sm ${
              message.success
                ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400"
                : "bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400"
            }`}>{message.text}</div>
          )}
        </div>
      )}

      <RunsTable runs={runs} configured={configured} fetchRuns={fetchRuns} />

      <div className="mt-6 bg-white dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 p-6">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-slate-200 mb-3">
          {tab === "data" ? "Data Pipeline Steps" : "Agent Pipeline Steps"}
        </h3>
        <div className="space-y-2 text-sm text-gray-600 dark:text-slate-400">
          {tab === "data" ? (
            <>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center text-xs font-bold">1</span> Scrape latest data from RBI &amp; NPCI</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center text-xs font-bold">2</span> Ingest &amp; process raw data into model-ready format</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center text-xs font-bold">3</span> Train aggregate + bank-level + txn volume models</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center text-xs font-bold">4</span> Sync forecasts &amp; historical data to Supabase</div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 flex items-center justify-center text-xs font-bold">1</span> Research — crawl news, earnings, policy sources</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 flex items-center justify-center text-xs font-bold">2</span> Extract — Claude-powered signal extraction from articles</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 flex items-center justify-center text-xs font-bold">3</span> Feature engineering — signals → regressors</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 flex items-center justify-center text-xs font-bold">4</span> Retrain — ensemble with new regressors, promote if better</div>
              <div className="flex items-center gap-3"><span className="w-6 h-6 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 flex items-center justify-center text-xs font-bold">5</span> Sync updated forecasts to Supabase</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
