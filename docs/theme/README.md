# Inference Docs Theme

This directory contains the custom theme for the Inference documentation site.

## Development Setup

1. Install Node.js dependencies
   ```bash
   npm install
   ```
2. Build the theme assets
   ```bash
   npm run build
   ```
3. Start the development server
   ```bash
   npm run dev
   ```

This will:
- Watch and compile Tailwind CSS
- Watch and bundle JavaScript with esbuild

3. In another terminal, start MkDocs:
```bash
# In the project root
mkdocs serve
```

## Homepage-Specific Assets

The Tailwind CSS and JavaScript are only loaded on the homepage. This is controlled in `main.html`:

```html
{% block extrahead %}
  {{ super() }}
  {% if page.is_homepage or (page.meta and page.meta.template == "home.html") %}
    <link href="{{ 'theme/assets/styles.css'|url }}" rel="stylesheet">
    <script type="module" src="{{ 'theme/assets/dist/home.js'|url }}"></script>
  {% endif %}
{% endblock %}
```

- `styles.css` - Tailwind styles for homepage components
- `home.js` - JavaScript for homepage animations and interactions

## Directory Structure

- `assets/` - Source files and compiled assets
  - `tailwind.css` - Tailwind source file (homepage only)
  - `home.js` - JavaScript source file (homepage only)
  - `dist/` - Compiled JavaScript (gitignored)
  - `styles.css` - Compiled CSS (gitignored)
  - `static/` - Static assets like images and animations
- `home.html` - Homepage template
- `main.html` - Base template that extends MkDocs Material

## Dependencies

- Tailwind CSS for homepage styling
- esbuild for JavaScript bundling
- GSAP for homepage animations
- Rive for interactive animations

## Build Process

The theme assets are automatically built when running `mkdocs build` through the build hook in `build.py`. This ensures the homepage assets are compiled before the site is built.