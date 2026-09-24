"""ffs predict command."""
import json

import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console
from ffs.predict_health import not_ready_reason, require_predictions, unusable_prediction_error


def _read_data_file(path: str):
    """Read CSV, JSON, or Parquet into a DataFrame."""
    import pandas as pd
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    elif lower.endswith(".json"):
        return pd.read_json(path)
    elif lower.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise click.ClickException(f"Unsupported file format (use .csv, .json, or .parquet): {path}")


def _get_predictor(state, model_id, target_column=None):
    """Get a predictor, optionally filtering by target column.

    Always through fm.list_predictors(). state.client.predictor() hands back a
    Predictor whose .id is the *session* id rather than the predictor's own, and
    the prediction endpoint 404s on that — so `ffs predict MODEL_ID RECORD`, the
    documented form with no --target-column, never reached a model at all.
    """
    fm = state.client.foundational_model(model_id)
    predictors = fm.list_predictors()

    if target_column:
        for p in predictors:
            if p.target_column == target_column:
                return p
        raise click.ClickException(
            f"No predictor found for target '{target_column}' on {model_id}"
        )

    if not predictors:
        raise click.ClickException(f"No predictor found in session {model_id}")
    if len(predictors) > 1:
        targets = ", ".join(p.target_column or p.id for p in predictors)
        raise click.ClickException(
            f"{model_id} has {len(predictors)} predictors ({targets}) — "
            f"choose one with --target-column"
        )
    return predictors[0]


@click.command()
@click.argument("model_id")
@click.argument("record_json", required=False)
@click.option("--file", "data_file", type=click.Path(exists=True), help="Batch predict from file (CSV, JSON, Parquet)")
@click.option("--target-column", default=None, help="Target column (if multiple predictors)")
@click.option("--explain", is_flag=True, help="Include feature importance")
@pass_client
def predict(state: ClientState, model_id, record_json, data_file, target_column, explain):
    """Make predictions.

    \b
    Single:  ffs predict MODEL_ID '{"col": "val"}'
    Batch:   ffs predict MODEL_ID --file data.csv
    """
    if not record_json and not data_file:
        raise click.ClickException("Provide a JSON record or --file")

    p = _get_predictor(state, model_id, target_column)

    # A predictor whose training job is queued, running or dead has no servable
    # model: the server answers with every field null rather than an error.
    reason = not_ready_reason(p)
    if reason:
        raise unusable_prediction_error(
            state, p.session_id, target_column=target_column or p.target_column, reason=reason,
        )
    target = target_column or p.target_column

    if data_file:
        df = _read_data_file(data_file)
        results = p.batch_predict(df)
        require_predictions(state, p.session_id, results, target_column=target)
        if state.output_json:
            print_json([r.to_dict() for r in results])
        else:
            console.print(f"[green]{len(results)} predictions[/green]\n")
            rows = []
            for i, r in enumerate(results):
                row = {"#": str(i + 1)}
                if r.predicted_class is not None:
                    row["Predicted"] = r.predicted_class
                elif r.prediction is not None:
                    row["Predicted"] = str(r.prediction)
                if r.confidence is not None:
                    row["Confidence"] = f"{r.confidence:.3f}"
                rows.append(row)
            if rows:
                print_list_table(rows, list(rows[0].keys()))
    else:
        record = json.loads(record_json)
        result = p.predict(record, feature_importance=explain)
        require_predictions(state, p.session_id, [result], target_column=target)
        if state.output_json:
            print_json(result.to_dict())
        else:
            data = {}
            if result.predicted_class is not None:
                data["Predicted"] = result.predicted_class
            elif result.prediction is not None:
                data["Predicted"] = str(result.prediction)
            if result.confidence is not None:
                data["Confidence"] = f"{result.confidence:.4f}"
            if result.probability is not None:
                data["Probability"] = f"{result.probability:.4f}"
            if result.probabilities:
                data["Distribution"] = "  ".join(f"{k}: {v:.3f}" for k, v in result.probabilities.items())
            if result.prediction_uuid:
                data["Prediction UUID"] = result.prediction_uuid
            if result.feature_importance:
                data["Feature Importance"] = ""
                print_kv(data, title="Prediction")
                for col, score in result.feature_importance.items():
                    console.print(f"  {col}: {score:.4f}")
            else:
                print_kv(data, title="Prediction")
