import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const GITHUB_REPO = process.env.GITHUB_REPO || "";
const GITHUB_PAT = process.env.GITHUB_PAT || "";

export async function POST(req: NextRequest) {
  if (!GITHUB_REPO || !GITHUB_PAT) {
    return NextResponse.json(
      { error: "GITHUB_REPO and GITHUB_PAT environment variables are required" },
      { status: 503 }
    );
  }

  const body = await req.json().catch(() => ({}));
  const skipScraping = body.skipScraping === true ? "true" : "false";
  const skipTraining = body.skipTraining === true ? "true" : "false";

  const res = await fetch(
    `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/monthly_pipeline.yml/dispatches`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${GITHUB_PAT}`,
        Accept: "application/vnd.github.v3+json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ref: "main",
        inputs: {
          skip_scraping: skipScraping,
          skip_training: skipTraining,
        },
      }),
    }
  );

  if (!res.ok) {
    const text = await res.text();
    return NextResponse.json(
      { error: `GitHub API error: ${res.status} ${text}` },
      { status: res.status }
    );
  }

  return NextResponse.json({ status: "dispatched", message: "Pipeline workflow triggered successfully" });
}

export async function GET() {
  if (!GITHUB_REPO || !GITHUB_PAT) {
    return NextResponse.json({ runs: [], configured: false });
  }

  const res = await fetch(
    `https://api.github.com/repos/${GITHUB_REPO}/actions/workflows/monthly_pipeline.yml/runs?per_page=5`,
    {
      headers: {
        Authorization: `Bearer ${GITHUB_PAT}`,
        Accept: "application/vnd.github.v3+json",
      },
    }
  );

  if (!res.ok) {
    return NextResponse.json({ runs: [], error: `GitHub API error: ${res.status}` });
  }

  const data = await res.json();
  const runs = (data.workflow_runs || []).map((r: any) => ({
    id: r.id,
    status: r.status,
    conclusion: r.conclusion,
    created_at: r.created_at,
    updated_at: r.updated_at,
    html_url: r.html_url,
    event: r.event,
    run_number: r.run_number,
  }));

  return NextResponse.json({ runs, configured: true });
}
