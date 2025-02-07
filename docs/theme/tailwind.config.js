import fluid, { extract } from "fluid-tailwind";

/** @type {import('tailwindcss').Config} */
export default {
  content: {
    files: [
      "./home.html",
      "./main.html",
      "./assets/**/*.{js,html}",
      "!./assets/dist/**",
      "!./node_modules/**"
    ],
    extract,
  },
  theme: {

    extend: {
      aspectRatio: {
        '700/312': '715 / 319',
        '593/230': '593 / 230',
        '580/205': '580 / 205',
        '548/230': '548 / 230',
      },
      // fontSize: {
      //   xs: "0.6rem",
      //   sm: "0.7rem",
      //   base: "0.8rem",
      //   lg: "0.9rem",
      //   xl: "1rem",
      //   "2xl": "1.2rem",
      //   "3xl": "1.5rem",
      //   "4xl": "1.8rem",
      //   "5xl": "2.4rem",
      //   "6xl": "3rem",
      //   "7xl": "3.6rem",
      // },
      fontSize: {
        xs: "12px", // 0.75rem
        sm: "14px", // 0.875rem
        base: "16px", // 1rem
        lg: "18px", // 1.125rem
        xl: "20px", // 1.25rem
        "2xl": "24px", // 1.5rem
        "3xl": "30px", // 1.875rem
        "4xl": "36px", // 2.25rem
        "5xl": "48px", // 3rem
        "6xl": "60px", // 3.75rem
        "7xl": "72px", // 4.5rem
      },
      // spacing: {
      //   px: "0.05rem",
      //   0: "0rem",
      //   1: "0.2rem",
      //   2: "0.4rem",
      //   3: "0.6rem",
      //   4: "0.8rem",
      //   5: "1rem",
      //   6: "1.2rem",
      //   8: "1.6rem",
      //   10: "2rem",
      //   12: "2.4rem",
      //   16: "3.2rem",
      //   20: "4rem",
      // },
      spacing: {
        px: "1px", // 1px
        0: "0px", // 0px
        1: "4px", // 0.25rem
        2: "8px", // 0.5rem
        3: "12px", // 0.75rem
        4: "16px", // 1rem
        5: "20px", // 1.25rem
        6: "24px", // 1.5rem
        8: "32px", // 2rem
        10: "40px", // 2.5rem
        12: "48px", // 3rem
        16: "64px", // 4rem
        20: "80px", // 5rem
      },
      // screens: {
      //   sm: "40rem",
      //   md: "51.2rem",
      //   lg: "64rem",
      //   xl: "80rem",
      //   "2xl": "96rem",
      // },
      screens: {
        sm: "640px", // 40rem
        md: "768px", // 51.2rem
        lg: "1024px", // 64rem
        xl: "1280px", // 80rem
        "2xl": "1536px", // 96rem
      },
    },
  },
  plugins: [fluid(), require('@tailwindcss/line-clamp')],
}