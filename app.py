# app.py
import os
import gradio as gr
import pandas as pd

from ecoeval.config import EcoEvalConfig
from ecoeval.datasets import load_dataset_by_name, list_available_datasets
from ecoeval.core import run_benchmark
from ecoeval.energy import run_with_energy
from ecoeval.logging_utils import append_run_to_csv, load_leaderboard


RUNS_CSV = "runs.csv"


def run_ecoeval(model_id: str, dataset_name: str, max_tasks: int):
    dataset = load_dataset_by_name(dataset_name)
    if max_tasks is not None and max_tasks > 0:
        max_tasks = min(max_tasks, len(dataset))
    else:
        max_tasks = len(dataset)

    cfg = EcoEvalConfig(
        model_id=model_id,
        max_new_tokens=128,
        temperature=0.2,
        top_p=0.95,
    )

    def bench_fn():
        return run_benchmark(dataset, cfg, limit=max_tasks)

    metrics = run_with_energy(bench_fn, project_name="EcoEval-LLM")

    # Build single-run summary table
    run_row = {
        "Model": model_id,
        "Dataset": dataset_name,
        "Tasks": metrics["tasks"],
        "Passed": metrics["passed"],
        "Accuracy": round(metrics["accuracy"], 3),
        "Runtime (s)": round(metrics["runtime_seconds"], 2),
        "Energy (kWh)": (
            round(metrics["energy_kwh"], 5) if metrics.get("energy_kwh") is not None else None
        ),
        "CO2eq (kg)": (
            round(metrics["emissions_kg"], 5) if metrics.get("emissions_kg") is not None else None
        ),
        "Energy / Task (kWh)": (
            round(metrics["energy_kwh"] / metrics["tasks"], 6)
            if metrics.get("energy_kwh") is not None and metrics["tasks"] > 0
            else None
        ),
        "CO2eq / Passed (kg)": (
            round(metrics["emissions_kg"] / metrics["passed"], 6)
            if metrics.get("emissions_kg") is not None and metrics["passed"] > 0
            else None
        ),
    }

    summary_df = pd.DataFrame([run_row])

    # Persist run to leaderboard CSV
    append_run_to_csv(RUNS_CSV, run_row)

    summary_text = (
        f"### Run summary\n"
        f"- **Model**: `{model_id}`\n"
        f"- **Dataset**: `{dataset_name}`\n"
        f"- **Tasks**: {metrics['tasks']}\n"
        f"- **Passed**: {metrics['passed']}  \n"
        f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
        f"- **Runtime**: {metrics['runtime_seconds']:.2f} s\n"
        f"- **Energy**: {metrics.get('energy_kwh', 'N/A')} kWh\n"
        f"- **CO₂eq**: {metrics.get('emissions_kg', 'N/A')} kg\n"
    )

    per_task_df = pd.DataFrame(metrics["per_task"])

    return summary_df, summary_text, per_task_df


def refresh_leaderboard():
    df = load_leaderboard(RUNS_CSV)
    if df is None or df.empty:
        return pd.DataFrame()
    # Sort by accuracy descending, then energy ascending
    sort_cols = []
    if "Accuracy" in df.columns:
        sort_cols.append("Accuracy")
    if "Energy (kWh)" in df.columns:
        sort_cols.append("Energy (kWh)")
    if sort_cols:
        df = df.sort_values(by=["Accuracy", "Energy (kWh)"], ascending=[False, True])
    return df.reset_index(drop=True)


def build_app():
    dataset_options = list_available_datasets()

    with gr.Blocks(title="EcoEval-LLM: Energy & Carbon Benchmarking for LLM Code Generation") as demo:
        gr.Markdown(
            """
# 🌱 EcoEval-LLM
Evaluate code generation models on **correctness**, **runtime**, **energy usage**, and **carbon emissions**.

This Space runs a small code-generation benchmark, executes unit tests, and tracks energy & CO₂ with [CodeCarbon](https://github.com/mlco2/codecarbon).
"""
        )

        with gr.Tab("Run Benchmark"):
            with gr.Row():
                model_in = gr.Textbox(
                    label="Model ID (Hugging Face Hub)",
                    value="Salesforce/codegen-350M-mono",
                    info="Any causal LM checkpoint that can generate Python code.",
                )
                dataset_in = gr.Dropdown(
                    choices=dataset_options,
                    value=dataset_options[0],
                    label="Dataset",
                )

            max_tasks_in = gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=5,
                label="Max tasks to evaluate",
                info="For heavy models, start small.",
            )

            run_btn = gr.Button("🚀 Run EcoEval Benchmark", variant="primary")

            gr.Markdown("### Run-level metrics")
            summary_table = gr.Dataframe(
                headers=[
                    "Model",
                    "Dataset",
                    "Tasks",
                    "Passed",
                    "Accuracy",
                    "Runtime (s)",
                    "Energy (kWh)",
                    "CO2eq (kg)",
                    "Energy / Task (kWh)",
                    "CO2eq / Passed (kg)",
                ],
                interactive=False,
                wrap=True,
            )

            summary_md = gr.Markdown()

            gr.Markdown("### Per-task results")
            per_task_table = gr.Dataframe(
                headers=[
                    "task_id",
                    "prompt_preview",
                    "passed",
                    "runtime_s",
                ],
                interactive=False,
                wrap=True,
            )

            run_btn.click(
                fn=run_ecoeval,
                inputs=[model_in, dataset_in, max_tasks_in],
                outputs=[summary_table, summary_md, per_task_table],
            )

        with gr.Tab("Leaderboard"):
            gr.Markdown(
                "Global history of runs in this Space (sorted by accuracy, then energy)."
            )
            refresh_btn = gr.Button("🔄 Refresh leaderboard")
            leaderboard_table = gr.Dataframe(interactive=False, wrap=True)

            refresh_btn.click(
                fn=refresh_leaderboard,
                inputs=None,
                outputs=leaderboard_table,
            )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch()
