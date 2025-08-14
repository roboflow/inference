import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#faf7ff',
          100: '#f3ecff', 
          200: '#e9d8ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#A351FB', // Main Roboflow brand purple
          600: '#8B5CF6',
          700: '#7C3AED',
          800: '#6B21A8',
          900: '#581C87',
        },
        accent: {
          purple: '#A351FB',
          'purple-light': '#8B5CF6',
          'purple-dark': '#7C3AED',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [
    require('@headlessui/tailwindcss')({ prefix: 'ui' })
  ],
}
export default config
