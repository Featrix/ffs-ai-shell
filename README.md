# ffs-ai-shell
The Featrix Foundation Shell (ffs)

Transform any CSV into a production-ready ML model from the command line.

## CLI Grammar

```
ffs [global-options] <command> <subcommand> [options] [args]
```

### Global Options
```
ffs --server URL          # API server (default: https://sphere-api.featrix.com)
ffs --cluster NAME        # Compute cluster (burrito, churro, etc.)
ffs --json                # Output raw JSON instead of formatted tables
ffs --quiet               # Minimal output
```

### Models (Foundational Models / Embedding Spaces)
```
ffs model create --name NAME --data FILE [--epochs N] [--ignore-columns COL,COL]
ffs model list [--prefix PREFIX]
ffs model show MODEL_ID
ffs model columns MODEL_ID
ffs model card MODEL_ID
ffs model wait MODEL_ID
ffs model extend MODEL_ID --data FILE [--epochs N]
ffs model encode MODEL_ID RECORD_JSON [--short]
ffs model publish MODEL_ID --org ORG --name NAME
ffs model unpublish MODEL_ID
ffs model deprecate MODEL_ID --message MSG --expires DATE
ffs model delete MODEL_ID
```

### Predictors
```
ffs predictor create MODEL_ID --target COLUMN --type {set,scalar} [--name NAME] [--data FILE]
ffs predictor list MODEL_ID
ffs predictor show MODEL_ID [--predictor-id ID]
ffs predictor metrics MODEL_ID [--predictor-id ID]
ffs predictor train-more MODEL_ID --epochs N [--predictor-id ID | --target COLUMN]
ffs predictor remove MODEL_ID {--predictor-id ID | --target COLUMN}
```

### Predict
```
ffs predict MODEL_ID RECORD_JSON [--target COLUMN] [--predictor-id ID]
ffs predict MODEL_ID --file FILE [--target COLUMN] [--sample N]
ffs predict explain MODEL_ID RECORD_JSON [--target COLUMN]
```

### Vector Database
```
ffs vectordb create MODEL_ID [--name NAME] [--records FILE]
ffs vectordb search MODEL_ID RECORD_JSON [-k N]
ffs vectordb add MODEL_ID --records FILE
ffs vectordb size MODEL_ID
```

### Server
```
ffs server health
```

### Usage Examples
```bash
# End-to-end: create model, train predictor, make prediction
ffs model create --name "customers" --data customers.csv
ffs model wait abc123
ffs predictor create abc123 --target churn --type set --rare-label "yes"
ffs model wait abc123
ffs predict abc123 '{"age": 35, "income": 50000}'

# Batch predict from CSV
ffs predict abc123 --file test_data.csv --target churn

# Similarity search
ffs vectordb create abc123 --name "customer_search"
ffs vectordb search abc123 '{"age": 35}' -k 10

# Pipe-friendly
ffs predict abc123 --file input.csv --json | jq '.predictions[].predicted_class'
```

## Architecture

- `MODEL_ID` = `session_id` in the underlying Featrix Sphere API
- CLI wraps the `featrixsphere` Python package
- Built with Click
