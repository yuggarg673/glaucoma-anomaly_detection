/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: { DEFAULT: "#1B6F6A", dark: "#0F4A47" },
        ink: { DEFAULT: "#14302F", soft: "#4A5F5D" },
        paper: "#F4F6F3",
        line: "#DCE3DE",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        serif: ['"Source Serif 4"', "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};
