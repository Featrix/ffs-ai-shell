# CLAUDE.md — ffs (Featrix Foundation Shell)

## Project overview
CLI tool (`ffs`) for building ML models via the Featrix Sphere API. Python, Click-based.

## Key files
- `ffs/cli.py` — main entry point, global options, login/whoami/upgrade commands
- `ffs/model_cmd.py` — `ffs foundation` subcommands (create, list, show, card, wait, extend, encode, publish, etc.)
- `ffs/predictor_cmd.py` — `ffs predictor` subcommands (create, list, show)
- `ffs/predict_cmd.py` — `ffs predict` command (single/batch predictions)
- `ffs/client.py` — ClientState, API key discovery, FeatrixSphere wrapper
- `ffs/output.py` — Rich-based output formatting (print_json, print_kv, print_list_table)
- `ffs/server_cmd.py` — server health check

## Architecture
- **CLI framework:** Click with `@click.group()` hierarchy and `@pass_client` decorator
- **API client:** `featrixsphere.FeatrixSphere` wraps the Sphere REST API
- **Output:** Rich console for terminal formatting; `--json` flag for raw JSON
- **Config:** `.featrix` files (JSON) with walk-up search from cwd to `~`
- **Constants:** `FEATRIX_UI = "https://featrix-ui.lovable.app"`, default API `https://sphere-api.featrix.com`

## Dependencies
- `click>=8.0`, `featrixsphere`, `rich`
- `pandas` used in predict_cmd for batch file I/O

## Related repos
- https://github.com/Featrix/model-card — model card renderers (Python, JS, React)
  - JS CDN: `https://bits.featrix.com/js/featrix-modelcard/model-card.js`
  - Use the CDN `<script>` approach for HTML rendering, NOT the Python `featrix-modelcard` package

## Publishing
- Package name on PyPI: `featrix-shell`
- Build: `python3 -m build`
- Upload: `python3 -m twine upload dist/*`
- Bump version in BOTH `pyproject.toml` and `ffs/__init__.py`

## Conventions
- Commands follow pattern: fetch object via `state.client.*()`, format with `print_json`/`print_kv`/`console.print`
- `state.output_json` controls JSON vs human-readable output
- Browser opens use `webbrowser.open()` with printed URL fallback
- When asked to publish: commit, push, build, and upload to PyPI — don't just edit files
