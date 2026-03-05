"""ffs train subcommands."""
import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, console


@click.group()
def train():
    """Train models on existing embedding spaces."""
    pass


@train.command()
@click.argument("es_session_id")
@click.option("--target-column", required=True, help="Column to predict")
@click.option("--type", "pred_type", required=True, type=click.Choice(["classifier", "regressor"]), help="Prediction type")
@click.option("--data", "data_file", required=True, type=click.Path(exists=True), help="Labels file (CSV/Parquet/JSON)")
@click.option("--name", default=None, help="Predictor name")
@click.option("--epochs", type=int, default=0, help="Training epochs (auto if 0)")
@pass_client
def model(state: ClientState, es_session_id, target_column, pred_type, data_file, name, epochs):
    """Train a predictor on an existing embedding space in a new session.

    Always creates a new session — does not modify the parent ES.

    \b
    Example:
      ffs train model ES_ID --target-column churn --type classifier --data labels.csv
    """
    fm = state.client.foundational_model(es_session_id)
    kwargs = dict(target_column=target_column, name=name, epochs=epochs, labels_file=data_file)

    if pred_type == "classifier":
        p = fm.create_binary_classifier(**kwargs)
    else:
        p = fm.create_regressor(**kwargs)

    if state.output_json:
        print_json({"predictor_id": p.id, "session_id": p.session_id,
                     "es_session_id": es_session_id, "target_column": p.target_column,
                     "type": pred_type, "status": p.status})
    else:
        console.print(f"[green]Predictor created:[/green] {p.id}")
        console.print(f"Session: {p.session_id}  (new)")
        console.print(f"ES: {es_session_id}")
        console.print(f"Target: {p.target_column}  Type: {pred_type}")
        console.print(f"Status: {p.status}")
        console.print(f"\nRun [bold]ffs foundation wait {p.session_id}[/bold] to monitor training.")
