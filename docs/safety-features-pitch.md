# Why ffs is Agent-Safe

Most ML CLIs are built for humans sitting at a terminal. `ffs` is built for both humans *and* autonomous AI agents that need to create, train, and deploy models without supervision. Here's what makes it safe to hand the keys to an agent.

## It speaks JSON natively

Every command has a `--json` mode. Not some commands — every one. An agent never has to regex-parse colored terminal output.

```bash
ffs --json foundation show $MODEL_ID | jq .status
```

## It fails fast and loud

A missing file, a bad predictor type, or a missing API key all fail *before* any API call is made. The exit code is always non-zero on failure. An agent doesn't waste 45 minutes of GPU training to discover a typo.

## It can't accidentally destroy things

The only destructive command (`foundation delete`) requires `--yes`. Everything else creates new resources or reads existing ones. `train model` always creates a new session — the parent embedding space is never touched.

## It never hangs

`foundation wait` has a configurable `--timeout` (default: 1 hour). On timeout or training failure, it exits non-zero. An agent can't get stuck in an infinite loop waiting for a model that errored out.

## Every project has its own identity

`ffs login` saves credentials to `.featrix` in the current directory. Each project gets its own API key, its own org, its own billing context. An agent working in `/projects/fraud-detection/` automatically uses that project's key — it can never accidentally hit the marketing team's account. `ffs whoami` confirms exactly which key is active and where it came from. Files are `chmod 600` on write.

If a `.featrix` file is tracked by git, `ffs` refuses to run and tells you exactly how to fix it — before your API key ends up in a remote repository.

## It works headless

Browser-open calls are wrapped in try/except. URLs are always printed to stdout. The login flow accepts `--api-key` directly. There's nothing in `ffs` that requires a display server, a browser, or interactive input (except `delete` without `--yes`).

## It shows its work

- `whoami` reports which API key is active and where it came from
- `server health` shows cluster status and versions
- `foundation wait` shows per-job progress, status, and duration — even for jobs not yet in the formal plan
- `--json` output always includes IDs and status fields that an agent can key on

## It self-upgrades safely

`ffs upgrade` uses the exact Python that's running it (`sys.executable`), works in virtualenvs, and treats individual package failures as non-fatal. An upgrade that fails for `featrixsphere` still attempts `featrix-shell`.

---

**Bottom line:** An agent can run `ffs` in a loop — creating models, polling for completion, making predictions — and the CLI will never hang, never silently corrupt data, never leak credentials across projects, and always give the agent a machine-readable signal about what happened.
