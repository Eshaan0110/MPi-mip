import { NextRequest, NextResponse } from "next/server";
import { supabase, isConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  if (!isConfigured) return NextResponse.json([], { status: 200 });

  const month = request.nextUrl.searchParams.get("month");
  const cardType = request.nextUrl.searchParams.get("card_type");
  const bank = request.nextUrl.searchParams.get("bank");

  let query = supabase.from("forecasts_bank").select("*").order("forecast_month", { ascending: true });
  if (month) query = query.eq("forecast_month", month);
  if (cardType) query = query.eq("card_type", cardType);
  if (bank) query = query.eq("bank_name", bank);

  const { data, error } = await query;
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  return NextResponse.json(data, {
    headers: { "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=1800" },
  });
}
