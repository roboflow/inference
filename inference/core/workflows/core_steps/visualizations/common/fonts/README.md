# Approved fonts

This package manages the approved fonts used for rich text rendering in
Workflows visualization blocks (e.g. `roboflow_core/rich_label_visualization@v1`,
which uses `supervision.RichLabelAnnotator` / Pillow).

Workflow authors select fonts by display name (e.g. `Geist Mono`; legacy
snake_case identifiers such as `geist_mono` are also accepted).
Names are resolved to local files by `resolve_font_path(...)` in
`__init__.py`. **Arbitrary filesystem paths and remote font URLs are
intentionally not supported** — only fonts registered in `registry.py` can be
used. This keeps rendering deterministic and avoids parsing untrusted font
files (FreeType/Pillow font parsing is a historical remote-code-execution
surface).

## Distribution model

Font assets are **not committed to git**. Only the registry code
(`registry.py`: pinned source and license URLs + SHA-256 checksums per font)
and the download logic live in the repository. Font binaries and license
texts (`assets/<identifier>/OFL.txt`) are provisioned by the shared script
`build_scripts/download_fonts.py`, which downloads each file from its pinned,
immutable URL and verifies it against the recorded checksum (hard failure on
mismatch).

How fonts reach each environment:

- **Docker images** — every server Dockerfile runs the download script at
  build time, baking verified fonts into the image, and sets
  `ALLOW_WORKFLOWS_FONTS_DOWNLOAD=False` (no runtime downloads; dev images
  keep it enabled because they typically mount local sources over
  `/app/inference`).
- **PyPI wheels** — `make create_wheels` depends on `make download_fonts`, so
  published wheels include the assets via `package_data`.
- **Local development** — run `make download_fonts` (idempotent; skips
  verified files). Test suites provision fonts automatically through the
  session-scoped `bundled_fonts` pytest fixture, which calls the same script.
- **Runtime fallback** — when a registered font is missing from the
  installation (e.g. source checkout without provisioning), it is downloaded
  on first use to `$MODEL_CACHE_DIR/workflows/fonts/`, checksum-verified,
  guarded by a file lock for multi-worker safety. The fallback is skipped —
  resolution fails fast with an actionable error instead — when
  `ALLOW_WORKFLOWS_FONTS_DOWNLOAD=False`, when `OFFLINE_MODE` is enabled, or
  when `SECURE_GATEWAY` is configured (the fonts' upstream hosts are not
  reachable through the gateway).

## Bundled fonts and attribution

All fonts are licensed under the [SIL Open Font License 1.1](https://openfontlicense.org).
The full license text for each font is provisioned (from a pinned,
checksum-verified URL) to `assets/<identifier>/OFL.txt` alongside the font
binary, as required by the OFL, and ships in Docker images and wheels. Fonts
are unmodified upstream builds (the OFL requires renaming modified versions;
we do not modify or subset fonts).

| Identifier | Family | Style | License | Version / pin | Size |
|---|---|---|---|---|---|
| `geist_mono` | Geist Mono (Vercel) | monospaced | OFL-1.1 | v1.7.2 | 146 KB |
| `jetbrains_mono` | JetBrains Mono | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 183 KB |
| `ibm_plex_mono` | IBM Plex Mono | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 132 KB |
| `source_code_pro` | Source Code Pro (Adobe) | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 207 KB |
| `space_mono` | Space Mono | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 97 KB |
| `roboto_mono` | Roboto Mono | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 179 KB |
| `geist` | Geist (Vercel) | proportional | OFL-1.1 | v1.7.2 | 123 KB |
| `inter` | Inter | proportional | OFL-1.1 | google/fonts@7ff85c87 | 856 KB |
| `open_sans` | Open Sans | proportional | OFL-1.1 | google/fonts@7ff85c87 | 520 KB |
| `noto_sans` | Noto Sans | proportional | OFL-1.1 | notofonts.github.io@eaa1a5cf | 607 KB |
| `fira_code` | Fira Code | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 254 KB |
| `inconsolata` | Inconsolata | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 339 KB |
| `anonymous_pro` | Anonymous Pro | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 154 KB |
| `pt_mono` | PT Mono | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 182 KB |
| `courier_prime` | Courier Prime | monospaced | OFL-1.1 | google/fonts@7ff85c87 | 70 KB |
| `roboto` | Roboto | proportional | OFL-1.1 | google/fonts@7ff85c87 | 477 KB |
| `lato` | Lato | proportional | OFL-1.1 | google/fonts@7ff85c87 | 641 KB |
| `montserrat` | Montserrat | proportional | OFL-1.1 | google/fonts@7ff85c87 | 727 KB |
| `work_sans` | Work Sans | proportional | OFL-1.1 | google/fonts@7ff85c87 | 353 KB |
| `nunito_sans` | Nunito Sans | proportional | OFL-1.1 | google/fonts@7ff85c87 | 558 KB |

Total provisioned size: ~6.5 MB.

Notes:

- Files named like `JetBrainsMono[wght].ttf` are variable fonts; Pillow loads
  them at their default (Regular) instance.
- `noto_sans` is the Latin/Greek/Cyrillic build of Noto Sans — it provides the
  broadest character coverage in the set, but does not cover CJK scripts.
  Glyphs missing from a selected font render as the font's `.notdef` glyph
  (typically an empty box); this is documented block behavior, not an error.

## Adding a new approved font (maintainers)

1. Verify the license permits redistribution bundled with software (OFL-1.1
   and Apache-2.0 qualify; include the full license text).
2. Pick an **immutable, version- or commit-pinned URL** for the font file
   (GitHub release tag or pinned commit of `google/fonts`). Never pin a
   mutable URL such as a `main` branch or a CDN endpoint.
3. Add a `FontMetadata` entry to `FONTS_REGISTRY` in `registry.py`, recording
   the pinned `source_url`, `license_url`, `version` and the SHA-256 checksums
   of both files (`shasum -a 256 <file>` on locally downloaded copies). Pin
   the license URL from the same repository/tag as the font binary. Nothing
   is committed under `assets/` — the whole directory is gitignored.
4. Run `make download_fonts` and confirm the new font and license download
   and verify.
5. The `font_family` dropdown in block manifests (e.g.
   `visualizations/rich_label/v1.py`) is generated from the registry — a unit
   test asserts the manifest enum matches the registry.
6. Update the table above.
7. Run `tests/workflows/unit_tests/core_steps/visualizations/test_fonts_registry.py`
   — it validates asset presence, checksums and license files.
