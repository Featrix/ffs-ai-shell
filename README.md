# ffs — The Featrix Foundation Shell

Transform any CSV into a production-ready ML predictor from the command line.

## Install

```bash
pip install featrix-shell
```

## Setup

```bash
ffs login            # save API key to ./.featrix (project-local)
ffs login --global   # save API key to ~/.featrix (user-wide)
ffs whoami           # verify identity and connection
ffs upgrade          # upgrade featrix-shell and featrixsphere
```

## Configuration

ffs looks for a `.featrix` file starting from the current directory and walking
up to `$HOME`. This lets you use different API keys per project:

```
~/work/client-a/.featrix    <- ffs uses this key when you're in client-a/
~/work/client-b/.featrix    <- ffs uses this key when you're in client-b/
~/.featrix                  <- fallback for everything else
```

Search order:
1. `FEATRIX_API_KEY` environment variable (always wins)
2. `.featrix` in current directory
3. `.featrix` in each parent directory up to `$HOME`
4. `~/.featrix`

The file is JSON:
```json
{"api_key": "sk_live_..."}
```

## CLI

```
ffs [global-options] <command> [subcommand] [options] [args]
```

### Global Options
```
--server URL          API server (default: https://sphere-api.featrix.com)
--cluster NAME        Compute cluster
--json                Output raw JSON
--quiet               Minimal output
```

### Authentication
```
ffs login [--global]                        Save API key (project-local or ~/.featrix)
ffs whoami                                  Show current user/org/connection
ffs upgrade                                 Upgrade featrix-shell and featrixsphere
```

### Foundation (Foundational Models / Embedding Spaces)
```
ffs foundation create --name NAME --data FILE [--epochs N] [--ignore-columns COL,COL]
ffs foundation list [--prefix PREFIX]
ffs foundation show MODEL_ID
ffs foundation columns MODEL_ID
ffs foundation card MODEL_ID
ffs foundation wait MODEL_ID [--poll-interval N] [--timeout N]
ffs foundation extend MODEL_ID --data FILE [--epochs N]
ffs foundation encode MODEL_ID RECORD_JSON [--short]
ffs foundation publish MODEL_ID --org ORG --name NAME
ffs foundation unpublish MODEL_ID
ffs foundation deprecate MODEL_ID --message MSG --expires DATE
ffs foundation cancel MODEL_ID --yes [--reason TEXT]
ffs foundation delete MODEL_ID
```

### Predictors
```
ffs predictor create MODEL_ID --target-column COL --type {classifier,regressor} [--labels FILE] [--name NAME] [--epochs N]
ffs predictor list MODEL_ID
ffs predictor show MODEL_ID
ffs predictor cancel MODEL_ID --yes [--reason TEXT]
```

### API Endpoints
```
ffs endpoint create MODEL_ID --name NAME [--api-key KEY] [--description TEXT]  Create a named endpoint for a predictor
ffs endpoint show MODEL_ID ENDPOINT_ID                                        Show endpoint details
ffs endpoint stats MODEL_ID ENDPOINT_ID                                       Show usage statistics
ffs endpoint regenerate-key MODEL_ID ENDPOINT_ID --yes                        Rotate the API key
ffs endpoint revoke-key MODEL_ID ENDPOINT_ID --yes                            Remove the API key (endpoint becomes public)
ffs endpoint delete MODEL_ID ENDPOINT_ID --yes                                Delete the endpoint
```

### Jobs
```
ffs jobs list [--prefix NAME]                                        List sessions with jobs queued/running, org-wide
ffs jobs cancel-queued [MODEL_ID] --yes [--prefix NAME] [--reason TEXT]  Cancel queued (not yet running) jobs
```

### PredictionNetworks
```
ffs network register NAME --spec-file FILE     Register or update a network's spec ({nodes, edges})
ffs network show NAME                          Show a registered network's spec
ffs network list                                List networks registered for your org
ffs network predict NAME RECORD_JSON            Run a network against one record
ffs network predict NAME --file FILE
```

### Predict
```
ffs predict MODEL_ID '{"col": "val"}'                          Single prediction (JSON)
ffs predict MODEL_ID --file FILE [--target-column COL]         Batch (CSV, JSON, Parquet)
ffs predict MODEL_ID '{"col": "val"}' --explain                Include feature importance
```

### Vector Database (not yet implemented)
```
ffs vectordb create MODEL_ID [--name NAME] [--records FILE]
ffs vectordb search MODEL_ID RECORD_JSON [-k N]
```

### Server
```
ffs server health
```

## Quick Start

```bash
# Login
ffs login

# Create a foundational model from CSV
ffs foundation create --name "customers" --data customers.csv

# Wait for foundation training
ffs foundation wait MODEL_ID

# Train a classifier on a target column
ffs predictor create MODEL_ID --target-column churned --type classifier

# Wait for predictor training (same wait command)
ffs foundation wait MODEL_ID

# Single prediction
ffs predict MODEL_ID '{"age": 35, "income": 50000}'

# Batch prediction from file
ffs predict MODEL_ID --file new_customers.csv
```

## Architecture

- `MODEL_ID` = `session_id` in the Featrix Sphere API
- Wraps the `featrixsphere` OO API (`FeatrixSphere`, `FoundationalModel`)
- Built with Click + Rich
