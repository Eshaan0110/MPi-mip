import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f0f5ff",
          100: "#e0ebff",
          500: "#2e75b6",
          600: "#1b5a9e",
          700: "#1b3a5c",
          900: "#0d1f33",
        },
        surface: {
          DEFAULT: "#1e293b",
          light: "#334155",
          dark: "#0f172a",
        },
      },
    },
  },
  plugins: [],
};

export default config;
