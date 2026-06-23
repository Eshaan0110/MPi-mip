import { isConfigured } from "@/lib/supabase";

export function ConfigWarning() {
  if (isConfigured) return null;
  return (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
      <p className="text-yellow-800 font-medium">Supabase not configured</p>
      <p className="text-yellow-700 text-sm mt-1">
        Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local to connect to your database.
      </p>
    </div>
  );
}
