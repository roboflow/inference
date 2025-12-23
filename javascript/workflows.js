
function render() {
    const colorList = [
        "#22c55e",
        "#14b8a6",
        "#ef4444",
        "#eab308",
        "#8b5cf6",
        "#f97316",
        "#3b82f6",
    ]

    const repoCards = document.querySelectorAll(".block-card");
    const labelsAll = Array
        .from(repoCards)
        .flatMap((element) => element.getAttribute('data-labels').split(','))
        .map(label => label.trim())
        .filter(label => label !== '');
    const uniqueLabels = [...new Set(labelsAll)];

    const labelToColor = uniqueLabels.reduce((map, label, index) => {
        map[label] = colorList[index % colorList.length];
        return map;
    }, {});

    async function setCard(el, url, name, desc, labels, theme, authors) {
      // console.log(name, desc)
      let labelHTML = ''
      if (labels) {
        const labelArray = labels.split(',').map((label, index) => {
          const color = labelToColor[label.trim()];
          return `<span class="non-selectable-text" style="background-color: ${color}; color: #fff; padding: 2px 4px; border-radius: 4px; margin-right: 4px;">${label}</span>`
        })

        labelHTML = labelArray.join(' ')
      }

      el.innerText = `
        <a style="text-decoration: none; color: inherit;" href="${url}">
          <div style="flex-direction: column; height: 100%; display: flex;
          font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; background: ${theme.background}; font-size: 14px; line-height: 1.5; color: ${theme.color}">
            <div style="display: flex; align-items: center;">
              <span style="font-weight: 700; font-size: 1rem; color: ${theme.linkColor};">
                ${name}
              </span>
            </div>
            <div style="font-size: 12px; margin-top: 0.5rem; margin-bottom: 0.5rem; color: ${theme.color}; flex: 1; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; height: calc(1.2em * 3);">
              ${desc}
            </div>
            <div style="font-size: 12px; color: ${theme.color}; display: flex; flex: 0; justify-content: space-between">
              <div style="display: flex; align-items: center;">
                ${labelHTML}
              </div>
            </div>
          </div>
        </a>
          `

      let sanitizedHTML = DOMPurify.sanitize(el.innerText);
      el.innerHTML = sanitizedHTML;
    }
    for (const el of document.querySelectorAll('.block-card')) {
      const url = el.getAttribute('data-url');
      const name = el.getAttribute('data-name');
      const desc = el.getAttribute('data-desc');
      const labels = el.getAttribute('data-labels');
      const authors = el.getAttribute('data-author');
      const palette = __md_get("__palette")
      if (palette && typeof palette.color === "object") {
        var theme = palette.color.scheme === "slate" ? "dark-theme" : "light-default"
      } else {
        var theme = "light-default"
      }

      setCard(el, url, name, desc, labels, theme, authors);
    }
};

document$.subscribe(render);
