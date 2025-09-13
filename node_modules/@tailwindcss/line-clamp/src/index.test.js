const path = require('path')
const postcss = require('postcss')
const tailwindcss = require('tailwindcss')
const lineClampPlugin = require('.')

function run(config, plugin = tailwindcss) {
  let { currentTestName } = expect.getState()
  config = {
    ...{ plugins: [lineClampPlugin], corePlugins: { preflight: false } },
    ...config,
  }

  return postcss(plugin(config)).process('@tailwind utilities', {
    from: `${path.resolve(__filename)}?test=${currentTestName}`,
  })
}

it('should add the `line-clamp-{n}` components', () => {
  const config = {
    content: [
      {
        raw: String.raw`<div class="line-clamp-2 line-clamp-[33] line-clamp-[var(--line-clamp-variable)]"></div>`,
      },
    ],
  }

  return run(config).then((result) => {
    expect(result.css).toMatchCss(String.raw`
      .line-clamp-2 {
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 2;
      }

      .line-clamp-\[33\] {
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: 33;
      }

      .line-clamp-\[var\(--line-clamp-variable\)\] {
        overflow: hidden;
        display: -webkit-box;
        -webkit-box-orient: vertical;
        -webkit-line-clamp: var(--line-clamp-variable);
      }
    `)
  })
})

it('should add the `line-clamp-none` utility', () => {
  const config = {
    content: [{ raw: String.raw`<div class="line-clamp-none"></div>` }],
  }

  return run(config).then((result) => {
    expect(result.css).toMatchCss(String.raw`
      .line-clamp-none {
        -webkit-line-clamp: unset;
      }
    `)
  })
})
