import pandas as pd, numpy as np, matplotlib.pyplot as plt, pathlib, sys

RESULTS_DIR = pathlib.Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IN = RESULTS_DIR / "pp_results_fast.csv"
if not IN.exists():
    print(f"Missing results CSV at {IN}.")
    sys.exit(1)

df = pd.read_csv(IN)
if df.empty:
    print(f"{IN} is empty (only header). Add runs, then re-try.")
    sys.exit(1)

if "timestamp" in df.columns:
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["schedule", "world_size", "layers", "heads"], keep="last"
    )

df["model"] = df["layers"].astype(str) + "L-" + df["heads"].astype(str) + "H"
df["Processes"] = df["world_size"]

base = (
    df[df["schedule"] == "gpipe"][["world_size", "layers", "heads", "tokens_per_s"]]
    .rename(columns={"tokens_per_s": "gpipe_tps"})
)

merged = df.merge(base, on=["world_size", "layers", "heads"], how="left")
merged = merged.dropna(subset=["gpipe_tps"]).copy()

merged["speedup_vs_gpipe"] = merged["tokens_per_s"] / merged["gpipe_tps"]
merged["scaling_efficiency_pct"] = (
    merged["speedup_vs_gpipe"] / merged["world_size"] * 100.0
)

cols = [
    "schedule", "world_size", "model", "layers", "heads",
    "tokens_per_s", "avg_step_time_s", "total_time_s",
    "speedup_vs_gpipe", "scaling_efficiency_pct"
]
table = merged[cols].sort_values(["world_size", "model", "schedule"])
print("\n=== Summary (all runs) ===")
print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

grouped = (
    merged.groupby(["layers", "heads", "Processes", "schedule"], as_index=False)
          .agg(**{
              "Tokens per Second": ("tokens_per_s", "mean"),
              "Average Step Time (s)": ("avg_step_time_s", "mean"),
              "Total Time (s)": ("total_time_s", "mean"),
              "Speedup vs GPipe": ("speedup_vs_gpipe", "mean"),
              "Scaling Efficiency (%)": ("scaling_efficiency_pct", "mean"),
          })
)
grouped["Model Size"] = grouped["layers"].astype(str) + "L-" + grouped["heads"].astype(str) + "H"
summary_csv = RESULTS_DIR / "pp_analysis_dedupe.csv"
grouped[[
    "layers","heads","Model Size","Processes","schedule",
    "Tokens per Second","Average Step Time (s)","Total Time (s)",
    "Speedup vs GPipe","Scaling Efficiency (%)"
]].sort_values(["layers","heads","Processes","schedule"]).to_csv(summary_csv, index=False)
print(f"\nWrote summary CSV: {summary_csv}")

def plot_metric(metric_col: str, title: str, ylabel: str, out_name: str):
    plot_df = merged[merged["schedule"].isin(["1f1b", "interleaved"])].copy()
    if plot_df.empty:
        print(f"Skipping plot {out_name}: no 1f1b/interleaved rows.")
        return
    pivot = plot_df.pivot_table(
        index="model",
        columns=["world_size", "schedule"],
        values=metric_col,
        aggfunc="mean",
    ).sort_index()
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Model Size")
    ax.set_ylabel(ylabel)
    ax.legend(title="Processes, schedule", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = RESULTS_DIR / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote plot: {out_path}")

plot_metric(
    metric_col="speedup_vs_gpipe",
    title="Speedup vs GPipe by Model Size and Process Count",
    ylabel="Speedup",
    out_name="speedup_vs_gpipe.png",
)

plot_metric(
    metric_col="scaling_efficiency_pct",
    title="Scaling Efficiency by Model Size and Process Count",
    ylabel="Scaling Efficiency (%)",
    out_name="scaling_efficiency.png",
)

print("\nDone.")
