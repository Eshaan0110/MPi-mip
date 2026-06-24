"use client";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[50vh] px-4">
      <h2 className="text-xl font-bold text-gray-800 mb-2">Something went wrong</h2>
      <p className="text-sm text-gray-500 mb-4">{error.message || "An unexpected error occurred."}</p>
      <button
        onClick={reset}
        className="px-4 py-2 bg-brand-600 text-white rounded-md text-sm hover:bg-brand-700 transition-colors"
      >
        Try again
      </button>
    </div>
  );
}
