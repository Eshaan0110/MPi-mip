import { isConfigured } from "@/lib/supabase";

export function ConfigWarning() {
  if (isConfigured) return null;
  return (
    <div className="bg-yellow-50 border border-yellow-200 dark:bg-yellow-900/30 dark:border-yellow-700/50 rounded-lg p-4 mb-6">
      <p className="text-yellow-800 dark:text-yellow-400 font-medium">Supabase not configured</p>
      <p className="text-yellow-700 dark:text-yellow-500 text-sm mt-1">
        Set NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY in .env.local (or your deployment&apos;s environment settings) to connect to your database.
      </p>
    </div>
  );
}
