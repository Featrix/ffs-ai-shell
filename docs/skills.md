# Claude Agent Skill: Featrix Foundation Shell (ffs)

How to use `ffs` to build production-ready ML predictors from CSV data.

## Setup

```bash
pip install ffs-ai-shell
# or from source:
git clone https://github.com/Featrix/ffs-ai-shell.git
cd ffs-ai-shell && pip install -e .
```

Verify:
```bash
ffs server health
```

## What Featrix Does

Featrix builds **neural embedding spaces** from tabular data, then trains **predictors** on top. The embedding space learns relationships across ALL your data (labeled + unlabeled). Predictors are trained on labeled subsets.

This separation is powerful:
- Embedding space gets rich context from all historical data
- Predictors focus on specific prediction tasks
- You don't need labels for everything

## Workflow

### Step 1: Create a Foundational Model

```bash
ffs model create --name "customer-churn" --data customers.csv
```

Output:
```
Model created: customer-churn-abc123-def456
Status: training

Run ffs model wait customer-churn-abc123-def456 to monitor training.
```

The `--name` becomes part of the model ID for easy identification.

### Step 2: Wait for Training

```bash
ffs model wait customer-churn-abc123-def456
```

Shows real-time progress:
```
Waiting for customer-churn-abc123-def456  (5m32s)
  done  create_structured_data (1s)
  done  pre_analysis_architecture (3s)
  running 67%  train_es [gpu_training] (5m28s)
  pending  train_knn
  pending  run_clustering
```

### Step 3: Train a Predictor

```bash
# Classification (predicting categories)
ffs predictor create customer-churn-abc123-def456 --target "churned" --type set

# Regression (predicting numbers)
ffs predictor create customer-churn-abc123-def456 --target "lifetime_value" --type scalar
```

### Step 4: Make Predictions

```bash
# Single prediction
ffs predict customer-churn-abc123-def456 '{"age": 35, "tenure_months": 24, "monthly_spend": 89.99}'

# Batch from file
ffs predict customer-churn-abc123-def456 --file new_customers.csv --json > predictions.json
```

## CLI Reference

### Global Options
```
--json            Output raw JSON for scripting
--quiet           Minimal output
```

### Model Commands
```bash
ffs model create --name NAME --data FILE [--epochs N] [--ignore-columns COL,COL]
ffs model show MODEL_ID
ffs model wait MODEL_ID [--poll-interval N] [--timeout N]
ffs model columns MODEL_ID          # List columns in embedding space
ffs model card MODEL_ID             # Full model metadata as JSON
ffs model encode MODEL_ID '{"col": "val"}'  # Get embedding vector
ffs model extend MODEL_ID --data FILE       # Add new data
ffs model publish MODEL_ID --org ORG --name NAME
ffs model unpublish MODEL_ID
ffs model delete MODEL_ID
```

### Predictor Commands
```bash
ffs predictor create MODEL_ID --target COLUMN --type {set,scalar} [--rare-label VALUE]
ffs predictor list MODEL_ID
ffs predictor show MODEL_ID [--predictor-id ID]
ffs predictor metrics MODEL_ID
```

### Prediction Commands
```bash
ffs predict MODEL_ID '{"feature": "value"}'
ffs predict MODEL_ID --file data.csv [--json]
ffs predict explain MODEL_ID '{"feature": "value"}'
```

## Common Agent Patterns

### Pattern 1: Quick Experiment
When user wants fast iteration:

```bash
# Fast training (fewer epochs)
ffs model create --name "experiment-v1" --data data.csv --epochs 50
MODEL_ID=$(ffs model create --name "exp" --data data.csv --epochs 50 --json | jq -r .model_id)
ffs model wait $MODEL_ID
ffs predictor create $MODEL_ID --target "outcome" --type set
ffs predict $MODEL_ID --file test.csv --json
```

### Pattern 2: Production Model
When user wants production quality:

```bash
# Full training (default ~250 epochs)
ffs model create --name "prod-v1" --data full_data.csv
ffs model wait $MODEL_ID  # May take 30-60 min for large datasets
ffs predictor create $MODEL_ID --target "outcome" --type set
ffs model publish $MODEL_ID --org "company" --name "predictor-v1"
```

### Pattern 3: Imbalanced Classification
When dealing with rare classes (fraud, churn, defects):

```bash
ffs predictor create $MODEL_ID --target "is_fraud" --type set --rare-label "fraud"
```

### Pattern 4: Scripted Pipeline
For automation:

```bash
#!/bin/bash
set -e

MODEL_ID=$(ffs model create --name "$1" --data "$2" --json | jq -r .model_id)
echo "Created model: $MODEL_ID"

ffs model wait $MODEL_ID --quiet
echo "Training complete"

ffs predictor create $MODEL_ID --target "$3" --type set --json
echo "Predictor ready"

ffs predict $MODEL_ID --file test.csv --json > predictions.json
echo "Predictions saved to predictions.json"
```

## Troubleshooting

### Check Status
```bash
ffs model show $MODEL_ID
ffs server health
```

### Training Failed
```bash
# wait command shows error details
ffs model wait $MODEL_ID
```

### View Model Details
```bash
# Column types, training params, metrics
ffs model card $MODEL_ID | jq .
```

## API Concepts

| Term | Meaning |
|------|---------|
| Model ID | Session ID in Featrix API |
| Foundational Model | Embedding space trained on your data |
| Predictor | Classifier/regressor on top of embedding space |
| Embedding | Vector representation of a record |

## Column Types

Featrix auto-detects:
- `scalar` — numeric (age, price, count)
- `set` — categorical (city, status, product_type)
- `free_string` — text (description, notes)

Override with `--ignore-columns` to exclude columns from training.
