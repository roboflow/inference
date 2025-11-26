const plugin = require('tailwindcss/plugin')

const baseStyles = {
  overflow: 'hidden',
  display: '-webkit-box',
  '-webkit-box-orient': 'vertical',
}

const lineClamp = plugin(
  function ({ matchUtilities, addUtilities, theme, variants }) {
    const values = theme('lineClamp')

    matchUtilities(
      {
        'line-clamp': (value) => ({
          ...baseStyles,
          '-webkit-line-clamp': `${value}`,
        }),
      },
      { values }
    )

    addUtilities(
      [
        {
          '.line-clamp-none': {
            '-webkit-line-clamp': 'unset',
          },
        },
      ],
      variants('lineClamp')
    )
  },
  {
    theme: {
      lineClamp: {
        1: '1',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
      },
    },
    variants: {
      lineClamp: ['responsive'],
    },
  }
)

module.exports = lineClamp
