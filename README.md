# ffs — The Featrix Foundation Shell

Transform any CSV into a production-ready ML predictor from the command line.

## Install

```bash
pip install featrix-shell
```

## Setup

```bash
ffs login          # prompts for API key, saves to ~/.featrix
ffs whoami         # verify identity and connection
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
ffs login                                   Save API key to ~/.featrix
ffs whoami                                  Show current user/org/connection
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
ffs foundation delete MODEL_ID
```

### Predictors (not yet implemented)
```
ffs predictor create MODEL_ID --target COLUMN --type {set,scalar}
ffs predictor list MODEL_ID
ffs predictor show PREDICTOR_ID
ffs predictor metrics PREDICTOR_ID
```

### Predict (not yet implemented)
```
ffs predict MODEL_ID RECORD_JSON [--target COLUMN]
ffs predict MODEL_ID --file FILE [--target COLUMN]
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

# Wait for training
ffs foundation wait customers-abc123

# Show model details
ffs foundation show customers-abc123

# Encode a record into the embedding space
ffs foundation encode customers-abc123 '{"age": 35, "income": 50000}'
```

## Architecture

- `MODEL_ID` = `session_id` in the Featrix Sphere API
- Wraps the `featrixsphere` OO API (`FeatrixSphere`, `FoundationalModel`)
- Built with Click + Rich
