import { NextResponse } from "next/server";
import { supabase, isConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!isConfigured) return NextResponse.json([], { status: 200 });

  const sources = ["rbi_bankwise", "rbi_psi", "npci_upi", "rbi_repo"];
  const statuses = [];

  for (const source of sources) {
    const { data } = await supabase
      .from("scraper_runs")
      .select("started_at, status")
      .eq("source", source)
      .order("started_at", { ascending: false })
      .limit(1)
      .single();

    statuses.push({ source, last_run: data?.started_at ?? null, status: data?.status ?? "never_run" });
  }

  return NextResponse.json(statuses, {
    headers: { "Cache-Control": "public, s-maxage=300, stale-while-revalidate=60" },
  });
}
