# Inference Docs Theme

This directory contains the custom theme for the Inference documentation site.

## Development Setup

Install Node.js dependencies
```bash
npm install
```

Then, start MkDocs:
```bash
# In the project root
python -m mkdocs serve
```

## Homepage-Specific Assets

The Tailwind CSS and JavaScript are only loaded on the homepage. This is controlled in `main.html`.

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