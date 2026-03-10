# ffs: Safe by Design for Humans and Agents

## The Problem

Building ML models typically requires chaining together API calls, managing credentials, uploading data, polling for training completion, and interpreting results. When a human does this manually, mistakes are annoying. When an AI agent does it autonomously, mistakes can be expensive — wasted compute, corrupted models, leaked credentials, or silent failures that go undetected.

`ffs` is designed so that both humans and autonomous agents can build, train, and deploy Featrix models safely. Every command follows the same principles: fail fast on bad input, never mutate silently, always produce parseable output, and make destructive actions hard to do accidentally.

---

## 1. Dual Output Modes: Human-Readable and Machine-Parseable

Every `ffs` command that produces output has two code paths:

```bash
# Human sees a formatted table
ffs foundation show MODEL_ID

# Agent gets structured JSON
ffs --json foundation show MODEL_ID
```

The `--json` flag is global — it works with every subcommand, not just some of them. Agents never need to scrape Rich-formatted terminal output. The JSON serializer uses `default=str` so non-standard types (datetimes, UUIDs) are coerced to strings rather than crashing.

A `--quiet` flag suppresses verbose output (e.g., the detail table after `wait` completes) while preserving exit codes. Agents that only care about success/failure can use `--quiet` to reduce noise.

---

## 2. Predictable Exit Codes

Every `ffs` command exits `0` on success and non-zero on failure. There are no ambiguous cases:

| Scenario | Exit Code |
|---|---|
| Command succeeds | 0 |
| Training completes | 0 |
| Training fails (error/failed status) | 1 |
| Wait times out | 1 |
| Bad input (missing file, wrong type) | Non-zero |
| API error (auth, network, server) | 1 |

The top-level exception handler catches all unhandled exceptions and prints a clean one-line `Error: <message>` instead of a Python traceback. Agents can rely on exit codes without parsing error output.

---

## 3. Fail-Fast Input Validation

Bad input is rejected *before* any API call is made:

- **File arguments** use `click.Path(exists=True)` — a missing data file produces `Error: Invalid value for '--data': Path 'missing.csv' does not exist.` without ever contacting the server.
- **Enum arguments** use `click.Choice` — `--type regression` fails immediately with the valid options listed.
- **Mutual dependencies** are checked explicitly — `ffs predict MODEL_ID` with neither a JSON record nor `--file` raises a clear error before attempting any API call.
- **Unsupported file formats** are caught at read time — `.xlsx` gets `Unsupported file format (use .csv, .json, or .parquet)` rather than a cryptic pandas error.

This fail-fast behavior is critical for agents: it means a bad invocation fails instantly and cheaply, rather than after uploading data or waiting for a training job.

---

## 4. Project-Scoped Credentials

### Every Project Gets Its Own API Key

`ffs login` saves credentials to `.featrix` in the current directory by default. This means each project or workspace has its own API key, its own Featrix identity, and its own billing context:

```
~/projects/fraud-detection/.featrix    ← Team A's key
~/projects/marketing-model/.featrix    ← Team B's key
~/.featrix                             ← Personal fallback
```

When an agent (or a human) runs `ffs` from within a project directory, it automatically picks up that project's credentials — no flags, no env vars, no configuration. An agent working on the fraud model can never accidentally bill to the marketing account or access the wrong organization's data.

For shared or global use, `ffs login --global` writes to `~/.featrix` instead.

### Layered Discovery with Clear Priority

API keys are resolved in a strict priority order:

1. `FEATRIX_API_KEY` environment variable (highest priority — for CI/CD and containers)
2. Project-local `.featrix` file (walk-up search from cwd toward `~`)
3. Global `~/.featrix` file (fallback)

The walk-up search mirrors `.gitconfig` / `.editorconfig` conventions. `ffs` walks from the current directory upward through parent directories, stopping at `~` or the filesystem root. This means a `.featrix` file at the repo root covers all subdirectories within that project.

### Git Tracking Protection

If `ffs` discovers that a `.featrix` file is tracked by git, it **refuses to run** and prints a clear error with exact remediation steps:

```
DANGER: /projects/fraud-detection/.featrix is tracked by git!

Your API key will be pushed to the remote repository.
Fix this now:

  git rm --cached /projects/fraud-detection/.featrix
  echo .featrix >> .gitignore
  git commit -m 'Remove .featrix from tracking'

Then rotate your API key at https://featrix-ui.lovable.app/api-keys
```

This prevents the most common credential leak vector: accidentally committing a config file that contains an API key. The check runs on every command, not just `login`.

### Hardened at Rest

`ffs login` writes the config file and immediately sets permissions to `0600` (owner read/write only). On shared systems, other users cannot read the API key.

### Transparent Source

`ffs whoami` reports where the active API key came from:

```
API Key Source: ./projects/fraud-detection/.featrix
```

or in JSON mode: `{"api_key_source": "FEATRIX_API_KEY env var"}`. An agent can verify it's using the expected credentials before proceeding. If credentials are being loaded from an unexpected location, this makes it immediately visible.

### Lazy Initialization

The API client is only constructed when a command actually needs it. `ffs --help`, `ffs completions`, and `ffs upgrade` all work without credentials. This means agents can discover the CLI's capabilities without authentication.

---

## 5. Destructive Operations Require Explicit Confirmation

The only destructive command — `ffs foundation delete` — requires interactive confirmation:

```
Are you sure you want to delete this model? [y/N]:
```

For non-interactive use (agents, scripts, CI), the `--yes` flag bypasses the prompt. An agent must explicitly pass `--yes`, making accidental deletion impossible through a typo or malformed command.

No other command mutates or deletes existing data. List, show, wait, card, whoami, and predict are all read-only. Create commands produce new resources rather than modifying existing ones.

---

## 6. Session Isolation: New Sessions, Never Parent Mutation

`ffs train model` trains a predictor on an existing embedding space but **always creates a new session**. The parent ES session is never modified. This is documented in the command's help text:

```
Always creates a new session — does not modify the parent ES.
```

This is a critical safety property for agents that may run multiple training experiments on the same foundation model. Each experiment gets its own session, and the foundation remains intact regardless of what happens during training.

---

## 7. Robust Async Monitoring

Training jobs are asynchronous. `ffs foundation wait` handles every edge case an agent might encounter:

### Timeout Protection

```bash
ffs foundation wait MODEL_ID --timeout 7200  # 2 hours max
```

Default timeout is 1 hour. After timeout, the command exits with code 1 — it never loops forever. Agents can set appropriate timeouts for their use case.

### Failure Detection

If training fails, the wait command exits 1 and prints per-job error details:

```
Training failed.
  train: OOM
```

### Graceful Handling of Incomplete State

When a session exists but no jobs have been scheduled yet (common immediately after creation), the wait command shows session-level status rather than displaying nothing:

```
js-render-v3-1885f6ba...  new  (12s)
  Session status: new
  Waiting for jobs to be scheduled...
```

Jobs that exist in the session but aren't yet in the formal job plan are also displayed. An agent always has visibility into what's happening.

### Configurable Poll Interval

```bash
ffs foundation wait MODEL_ID --poll-interval 30  # check every 30s
```

Agents that want to reduce API load can increase the poll interval.

---

## 8. Environment Variable Overrides for CI/CD

Both `--server` and `--cluster` accept environment variables (`FFS_SERVER`, `FFS_CLUSTER`). These options are hidden from `--help` to avoid confusing end users, but agents and CI pipelines can use them to target different API environments without modifying config files or passing CLI flags.

---

## 9. Self-Upgrading Without Breaking

`ffs upgrade` upgrades both `featrix-shell` and `featrixsphere` packages:

- Uses `sys.executable` (the Python running ffs) rather than a bare `pip`, so it works correctly in virtualenvs and conda environments.
- A failure upgrading one package does not prevent upgrading the other.
- Upgrade failures are reported but do not cause a non-zero exit — the CLI remains usable.
- `--break-system-packages` is available (hidden) for system-level Python installations.

---

## 10. Tab Completion Reduces Agent and Human Errors

`ffs completions` outputs a shell completion script that enables tab completion for all commands, subcommands, and options. When completion isn't installed, `ffs` (with no subcommand) displays a setup tip:

```
Tip: Enable tab completion for ffs:
  echo 'eval "$(ffs completions)"' >> ~/.bashrc
```

The detection is read-only — `ffs` never writes to shell rc files automatically.

---

## Design Principles Summary

| Principle | How ffs Implements It |
|---|---|
| **Parseable output** | Global `--json` flag on every command |
| **Predictable exit codes** | 0 = success, non-zero = failure, always |
| **Fail fast** | Input validated before any API call |
| **No silent mutation** | Destructive ops require `--yes`; train creates new sessions |
| **Credential isolation** | Per-project `.featrix` files, walk-up discovery |
| **Credential security** | `chmod 600`, env var priority, source transparency |
| **Timeout protection** | `wait --timeout` with non-zero exit on timeout |
| **Graceful degradation** | Missing jobs, bad timestamps, browser failures all handled |
| **Environment flexibility** | Env vars for server, cluster, API key |
| **Self-diagnosing** | `whoami` shows key source; `server health` shows cluster state |
