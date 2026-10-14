"""
Summarizes the cross-validation results of an nnU-Net v2 model (validation/summary.json of every fold).

Outputs Markdown tables with mean/median Dice and IoU per fold and class, along with the
worst-performing cases for annotation review, and saves the report to results/cv_summary.md.

Usage:
    uv run python -m src.utils.summarize
    uv run python -m src.utils.summarize monai.runner.trainer_class_name=nnUNetTrainerMetrics_250epochs
    uv run python -m src.utils.summarize monai.visualize.worst_k=5
"""
import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

import src.utils.resolvers  # noqa: F401  (registers the OmegaConf resolvers used in cfg/)

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
log = logging.getLogger(__name__)


def get_class_names(model_dir: Path, raw_dir: Path, keys: list[int]) -> dict[int, str]:
    """Maps integer class keys (1, 2, ...) to names from dataset.json."""
    for candidate in (model_dir / "dataset.json", raw_dir / "dataset.json"):
        if candidate.is_file():
            labels = json.loads(candidate.read_text(encoding="utf-8")).get("labels", {})
            value_to_name = {int(v): name for name, v in labels.items() if int(v) != 0}
            return {k: value_to_name.get(k, str(k)) for k in keys}
    return {k: str(k) for k in keys}


def load_metrics_to_dataframe(summary_files: dict[str, Path]) -> pd.DataFrame:
    """Loads fold validation summary.json files into a single tidy DataFrame."""
    records = []
    for fold, path in summary_files.items():
        summary = json.loads(path.read_text(encoding="utf-8"))
        for case in summary.get("metric_per_case", []):
            case_id = Path(case["reference_file"]).stem
            for class_key, metrics in case["metrics"].items():
                records.append({
                    "fold": fold,
                    "case": case_id,
                    "class": int(class_key),
                    "Dice": metrics.get("Dice"),
                    "IoU": metrics.get("IoU"),
                })
    return pd.DataFrame(records)


def fmt_mean_sd(series: pd.Series) -> str:
    """Formats fold-level mean +/- SD (or mean alone if only one fold is present)."""
    mean = series.mean()
    return f"{mean:.3f} ± {series.std():.3f}" if len(series) > 1 else f"{mean:.3f}"


def generate_metric_table(df: pd.DataFrame, class_names: dict[int, str], metric: str) -> str:
    """Generates a Markdown table for a specific metric (Dice or IoU)."""
    valid = df.dropna(subset=[metric])
    class_keys = sorted(class_names)

    # Mean and median per fold and class
    agg = valid.groupby(["fold", "class"])[metric].agg(["mean", "median"])
    agg["formatted"] = agg.apply(lambda r: f"{r['mean']:.3f} / {r['median']:.3f}", axis=1)

    pivot = (
        agg.reset_index()
        .pivot(index="fold", columns="class", values="formatted")
        .reindex(columns=class_keys)
    )
    pivot.columns = [class_names[k] for k in class_keys]

    # Case count (n) and foreground mean/median per fold
    pivot.insert(0, "n", df.groupby("fold")["case"].nunique().astype(str))
    fg_per_case = valid.groupby(["fold", "case"])[metric].mean()
    fg_stats = fg_per_case.groupby("fold").agg(["mean", "median"])
    pivot["Foreground"] = fg_stats.apply(lambda r: f"{r['mean']:.3f} / {r['median']:.3f}", axis=1)

    # Mean +/- SD across folds
    fold_means = valid.groupby(["fold", "class"])[metric].mean().unstack().reindex(columns=class_keys)
    fg_fold_means = fg_per_case.groupby("fold").mean()

    mean_sd_row = {"n": ""}
    for k in class_keys:
        mean_sd_row[class_names[k]] = fmt_mean_sd(fold_means[k].dropna())
    mean_sd_row["Foreground"] = fmt_mean_sd(fg_fold_means.dropna())

    pivot.loc["**Mean ± SD (folds)**"] = mean_sd_row
    pivot.index.name = "Fold"

    table = pivot.reset_index().to_markdown(
        index=False,
        colalign=("left", "right") + ("center",) * (len(pivot.columns) - 1),
        disable_numparse=True,
    )
    return f"### {metric} (mean / median over validation cases)\n\n{table}\n"


def generate_worst_cases(df: pd.DataFrame, class_names: dict[int, str], n: int) -> str:
    """Finds the lowest-Dice cases per class across all folds."""
    valid = df.dropna(subset=["Dice"])
    rows = []
    for class_key, name in sorted(class_names.items()):
        worst = valid[valid["class"] == class_key].nsmallest(n, "Dice")
        cases_str = ", ".join(f"{r['case']} ({r['fold']}, {r['Dice']:.3f})" for _, r in worst.iterrows())
        rows.append({"Class": name, "Cases (fold, Dice)": cases_str})

    table = pd.DataFrame(rows).to_markdown(index=False, colalign=("left", "left"), disable_numparse=True)
    return f"### Lowest Dice cases (top {n} per class)\n\n{table}\n"


@hydra.main(version_base="1.3", config_path="../../cfg", config_name="config")
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.monai.visualize.model_dir)
    raw_dir = Path(cfg.paths.nnunet_raw)
    results_dir = Path(cfg.paths.results)
    worst_k = int(cfg.monai.visualize.get("worst_k", 3))

    folds = [f if str(f).startswith("fold_") else f"fold_{f}" for f in cfg.monai.visualize.folds]
    summary_files = {
        fold: model_dir / fold / "validation" / "summary.json"
        for fold in folds
        if (model_dir / fold / "validation" / "summary.json").is_file()
    }

    if not summary_files:
        log.warning(f"No fold_*/validation/summary.json found in {model_dir}")
        return

    df = load_metrics_to_dataframe(summary_files)
    class_keys = sorted(df["class"].unique())
    class_names = get_class_names(model_dir, raw_dir, class_keys)

    sections = [f"## {model_dir.name}\n"]
    for metric in ("Dice", "IoU"):
        sections.append(generate_metric_table(df, class_names, metric))
    if worst_k > 0:
        sections.append(generate_worst_cases(df, class_names, worst_k))

    report = "\n".join(sections)
    print(report)

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "cv_summary.md"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Saved summary report to {report_path}")


if __name__ == "__main__":
    main()
