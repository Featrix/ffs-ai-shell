# ffs Agent-Safety Checklist

## Output & Parsing
- [x] Global `--json` flag for machine-readable output on all commands
- [x] Global `--quiet` flag to suppress verbose output
- [x] `json.dumps(default=str)` prevents serialization crashes
- [x] Consistent exit codes: 0 = success, non-zero = failure

## Input Validation (fail before API calls)
- [x] `click.Path(exists=True)` on all file arguments
- [x] `click.Choice` on enum arguments (classifier/regressor, bash/zsh/fish)
- [x] Explicit mutual-dependency checks (predict requires record or --file)
- [x] Unsupported file formats rejected with clear message

## Credential Safety
- [x] Env var `FEATRIX_API_KEY` takes priority over config files
- [x] Walk-up `.featrix` discovery (project-local isolation)
- [x] Per-project credentials: each workspace gets its own API key
- [x] Git tracking detection: refuses to run if `.featrix` is tracked by git
- [x] Config files `chmod 600` on write
- [x] `whoami` reports key source for verification
- [x] Lazy client init — `--help` works without credentials
- [x] Login merges into existing config (doesn't overwrite other keys)
- [x] Accepts both `api_key` and `featrix_api_key` config key names

## Mutation Safety
- [x] `foundation delete` requires `--yes` for non-interactive use
- [x] `train model` always creates new session, never modifies parent ES
- [x] No command modifies existing models/sessions in place
- [x] All list/show/wait/card/predict/whoami commands are read-only

## Async / Long-Running Safety
- [x] `foundation wait --timeout` (default 3600s, configurable)
- [x] Timeout exits non-zero (never hangs)
- [x] Training failure exits non-zero with per-job error details
- [x] Handles "no jobs scheduled yet" state gracefully
- [x] Shows jobs not yet in formal plan
- [x] Configurable `--poll-interval`

## Headless / CI Compatibility
- [x] `FFS_SERVER` and `FFS_CLUSTER` env vars for environment targeting
- [x] `--api-key` flag on login bypasses browser flow
- [x] All `webbrowser.open()` calls wrapped in try/except
- [x] URLs always printed before browser open attempt
- [x] No interactive input required (except delete without --yes)

## Self-Maintenance
- [x] `upgrade` uses `sys.executable` (correct in virtualenvs)
- [x] Package upgrade failures are non-fatal
- [x] `--break-system-packages` available for system Python
- [x] `completions` command for tab completion setup
- [x] Completion tip shown only when not already installed

## Error Handling
- [x] Top-level exception handler — no raw tracebacks
- [x] Clean `Error: <message>` format on all failures
- [x] Defensive datetime parsing (malformed timestamps don't crash)
- [x] Soft failure on login verification (key saved even if health check fails)
- [x] `ClickException` preserves original exit codes
