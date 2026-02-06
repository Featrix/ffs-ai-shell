# Claude Skill: Featrix Foundation Shell (ffs)

Build production-ready ML predictors from CSV data in minutes.

## What Featrix Does

Featrix builds **neural embedding spaces** from your data, then trains **predictors** on top of them. The embedding space learns the relationships in your data; predictors classify or regress on specific target columns.

Key insight: The embedding space is trained on ALL your data (labeled + unlabeled), giving it rich context. Predictors are trained on labeled subsets. This separation is powerful — you get the benefit of all your historical data without needing labels for everything.

## Quick Start

```bash
# 1. Create a foundational model from your CSV
ffs model create --name "my-model" --data data.csv

# 2. Wait for embedding space training
ffs model wait <model_id>

# 3. Train a predictor on a target column
ffs predictor create <model_id> --target "outcome" --type set

# 4. Make predictions
ffs predict <model_id> '{"feature1": "value1", "feature2": 42}'
```

## CLI Reference

### Global Options
- `--server URL` — API endpoint (default: https://sphere-api.featrix.com)
- `--cluster NAME` — Compute cluster (burrito, churro)
- `--json` — Output raw JSON for scripting
- `--quiet` — Minimal output

### Model Commands
```bash
ffs model create --name NAME --data FILE [--epochs N] [--ignore-columns COL,COL]
ffs model show MODEL_ID
ffs model wait MODEL_ID [--poll-interval N] [--timeout N]
ffs model columns MODEL_ID
ffs model card MODEL_ID
ffs model encode MODEL_ID '{"col": "value"}' [--short]
ffs model delete MODEL_ID
```

### Predictor Commands
```bash
ffs predictor create MODEL_ID --target COLUMN --type {set,scalar} [--name NAME]
ffs predictor list MODEL_ID
ffs predictor show MODEL_ID [--predictor-id ID]
ffs predictor metrics MODEL_ID
```

### Prediction Commands
```bash
ffs predict MODEL_ID '{"feature": "value"}'
ffs predict MODEL_ID --file test.csv
ffs predict explain MODEL_ID '{"feature": "value"}'
```

## Building Rock-Solid Predictors

### 1. Data Preparation

Featrix handles messy real-world data well, but some prep helps:

```bash
# Check what columns are in your data
head -1 data.csv

# Create model — Featrix auto-detects column types
ffs model create --name "customer-churn" --data customers.csv
```

**Column types** (auto-detected):
- `scalar` — numeric values (age, price, count)
- `set` — categorical values (city, product_type, status)
- `free_string` — text fields (description, notes)

### 2. Training Strategy

**For classification** (predicting categories):
```bash
ffs predictor create $MODEL_ID --target "churn" --type set
```

**For regression** (predicting numbers):
```bash
ffs predictor create $MODEL_ID --target "price" --type scalar
```

**Handling imbalanced classes** (e.g., 95% good / 5% fraud):
```bash
# The API handles this — specify the rare class
ffs predictor create $MODEL_ID --target "is_fraud" --type set --rare-label "fraud"
```

### 3. Evaluating Models

```bash
# Get training metrics
ffs predictor metrics $MODEL_ID

# Test on held-out data
ffs predict $MODEL_ID --file test_data.csv

# Understand individual predictions
ffs predict explain $MODEL_ID '{"age": 35, "income": 75000}'
```

### 4. Production Deployment

Once training completes, the model is immediately available for predictions:

```bash
# Single prediction
ffs predict $MODEL_ID '{"feature1": "val1", "feature2": 123}'

# Batch predictions from file
ffs predict $MODEL_ID --file new_customers.csv --json > predictions.json
```

## Common Patterns

### Pattern 1: Quick Iteration
```bash
# Fast training for experimentation (fewer epochs)
ffs model create --name "experiment-v1" --data data.csv --epochs 50
ffs model wait $MODEL_ID
ffs predictor create $MODEL_ID --target "outcome" --type set
```

### Pattern 2: Production Model
```bash
# Full training for production (default epochs, ~250)
ffs model create --name "prod-v1" --data full_historical_data.csv
ffs model wait $MODEL_ID  # May take 30-60 min for large datasets
ffs predictor create $MODEL_ID --target "outcome" --type set
ffs model publish $MODEL_ID --org "mycompany" --name "customer-predictor"
```

### Pattern 3: Extending with New Data
```bash
# Add new data to existing model
ffs model extend $MODEL_ID --data new_data.csv
ffs model wait $NEW_MODEL_ID
```

## Monitoring Training

The `wait` command shows real-time job progress:

```
Waiting for credit-model-abc123  (2m15s)
  done  create_structured_data (1s)
  done  pre_analysis_architecture (3s)
  running 45%  train_es [gpu_training] (2m11s)
  pending  train_knn
  pending  run_clustering
```

Jobs in the pipeline:
- `create_structured_data` — Parses and encodes your CSV
- `pre_analysis_architecture` — Analyzes data complexity, picks network depth
- `train_es` — Trains the embedding space (main training, GPU)
- `train_knn` — Builds fast lookup structures
- `run_clustering` — Identifies natural groupings

## Tips

1. **Name your models descriptively** — The name becomes part of the model ID
2. **Use `--json` for scripting** — Parse output with jq
3. **Check `ffs model card`** — Full model metadata including column types, training params
4. **Predictions include confidence** — Use this to flag uncertain cases for human review

## When Things Go Wrong

```bash
# Check model status
ffs model show $MODEL_ID

# View detailed job errors
ffs model wait $MODEL_ID  # Will show error details if training failed

# Server health
ffs server health
```

## API Concepts

- **Model ID** = Session ID in the Featrix API
- **Foundational Model** = Embedding space trained on your data
- **Predictor** = Classifier or regressor trained on top of the embedding space
- **Embedding** = Vector representation of a record in the learned space
