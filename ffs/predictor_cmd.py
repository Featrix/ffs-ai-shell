"""ffs predictor subcommands."""
import click

from ffs.client import pass_client, ClientState
from ffs.output import print_json, print_kv, print_list_table, console


@click.group()
def predictor():
    """Train and manage predictors."""
    pass


@predictor.command()
@click.argument("model_id")
@click.option("--target-column", required=True, help="Column to predict")
@click.option("--type", "pred_type", required=True, type=click.Choice(["classifier", "regressor"]), help="Prediction type")
@click.option("--labels", "labels_file", default=None, type=click.Path(exists=True), help="Separate labels file (CSV/JSON/Parquet)")
@click.option("--name", default=None, help="Predictor name")
@click.option("--epochs", type=int, default=0, help="Training epochs (auto if 0)")
@pass_client
def create(state: ClientState, model_id, target_column, pred_type, labels_file, name, epochs):
    """Train a predictor on a foundation.

    If --labels is given, the labels are joined to the foundation's data
    for training (creates a new session).
    """
    fm = state.client.foundational_model(model_id)
    kwargs = dict(target_column=target_column, name=name, epochs=epochs)
    if labels_file:
        kwargs["labels_file"] = labels_file

    if pred_type == "classifier":
        p = fm.create_binary_classifier(**kwargs)
    else:
        p = fm.create_regressor(**kwargs)

    if state.output_json:
        print_json({"predictor_id": p.id, "session_id": p.session_id,
                     "target_column": p.target_column, "type": pred_type, "status": p.status})
    else:
        console.print(f"[green]Predictor created:[/green] {p.id}")
        console.print(f"Target: {p.target_column}  Type: {pred_type}")
        console.print(f"Status: {p.status}")
        console.print(f"\nRun [bold]ffs foundation wait {p.session_id}[/bold] to monitor training.")


@predictor.command("list")
@click.argument("model_id")
@pass_client
def list_predictors(state: ClientState, model_id):
    """List predictors for a foundation."""
    fm = state.client.foundational_model(model_id)
    predictors = fm.list_predictors()
    if state.output_json:
        print_json([p.to_dict() for p in predictors])
    elif not predictors:
        console.print("No predictors found.")
    else:
        rows = []
        for p in predictors:
            rows.append({
                "ID": p.id or "—",
                "Target": p.target_column,
                "Type": p.target_type,
                "Status": p.status or "—",
                "Accuracy": f"{p.accuracy:.4f}" if p.accuracy else "—",
            })
        print_list_table(rows, ["ID", "Target", "Type", "Status", "Accuracy"])


@predictor.command()
@click.argument("model_id")
@pass_client
def show(state: ClientState, model_id):
    """Show predictor details and metrics."""
    p = state.client.predictor(model_id)

    if state.output_json:
        print_json(p.to_dict())
    else:
        data = {
            "Predictor ID": p.id or "—",
            "Session ID": p.session_id,
            "Target": p.target_column,
            "Type": p.target_type,
            "Status": p.status or "—",
        }
        if p.accuracy is not None:
            data["Accuracy"] = f"{p.accuracy:.4f}"
        if p.auc is not None:
            data["AUC"] = f"{p.auc:.4f}"
        if p.f1 is not None:
            data["F1"] = f"{p.f1:.4f}"
        print_kv(data, title="Predictor")
