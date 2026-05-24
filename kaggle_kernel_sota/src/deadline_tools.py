from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Iterable


DEADLINE_RUN_TAG = "deadline_sweep"


def build_deadline_run_plan() -> list[dict]:
    """Return the small, Kaggle-only sweep plan for the final deadline run."""
    base = {
        "run_tag": DEADLINE_RUN_TAG,
        "batch_size": 32,
        "grad_clip": 1.0,
        "scheduler_name": "cosine",
        "seed": 42,
        "use_class_weight": False,
    }
    runs = [
        {
            "dataset": "BCIC2A",
            "model": "ShallowFBCSPNet",
            "epochs": 200,
            "lr": 3e-4,
            "patience": 40,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
        },
        {
            "dataset": "MDD",
            "model": "Deep4Net",
            "epochs": 80,
            "lr": 3e-4,
            "patience": 20,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
        },
        {
            "dataset": "SEED",
            "model": "Deep4Net",
            "epochs": 100,
            "lr": 3e-4,
            "patience": 25,
            "weight_decay": 1e-3,
            "label_smoothing": 0.05,
        },
        {
            "dataset": "SLEEP",
            "model": "USleep",
            "epochs": 100,
            "lr": 5e-4,
            "patience": 20,
            "weight_decay": 1e-4,
            "label_smoothing": 0.02,
            "use_class_weight": True,
        },
    ]
    return [{**base, **run} for run in runs]


def validate_prediction_file(
    pred_path: str | Path,
    expected_count: int | None = None,
    num_classes: int | None = None,
) -> list[str]:
    """Validate one course-format prediction file and return human-readable errors."""
    path = Path(pred_path)
    if not path.exists():
        return [f"missing prediction file: {path}"]

    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    if expected_count is not None and len(lines) != expected_count:
        errors.append(f"expected {expected_count} predictions, found {len(lines)}")

    for idx, raw in enumerate(lines, start=1):
        value = raw.strip()
        try:
            label = int(value)
        except ValueError:
            errors.append(f"line {idx} is not an integer: {value}")
            continue
        if num_classes is not None and not 0 <= label < num_classes:
            errors.append(f"line {idx} label {label} outside [0, {num_classes - 1}]")
    return errors


def select_best_runs(
    summaries: Iterable[Iterable[dict]],
    min_improvement: float = 0.0,
) -> dict[str, dict]:
    """Select the best valid run per dataset, keeping formal as a reliable fallback."""
    selected: dict[str, dict] = {}

    for summary in summaries:
        for run in summary:
            if run.get("status") == "skipped":
                continue
            dataset = run.get("dataset")
            if not dataset:
                continue
            try:
                score = float(run.get("best_val_acc"))
            except (TypeError, ValueError):
                continue
            exp_dir = run.get("experiment_dir")
            if not exp_dir or not (Path(exp_dir) / "predictions.txt").exists():
                continue

            current = selected.get(dataset)
            if current is None:
                selected[dataset] = dict(run)
                continue

            current_score = float(current["best_val_acc"])
            if score > current_score + min_improvement:
                selected[dataset] = dict(run)

    return selected


def install_packaged_formal_fallback(package_dir: str | Path, results_dir: str | Path) -> bool:
    """Copy packaged formal predictions into the active Kaggle results directory."""
    package = Path(package_dir)
    target = Path(results_dir)
    summary_path = package / "summary_formal.json"
    if not package.exists() or not summary_path.exists():
        return False

    target.mkdir(parents=True, exist_ok=True)
    for dataset_dir in package.iterdir():
        if dataset_dir.is_dir():
            dst_dataset = target / dataset_dir.name
            if dst_dataset.exists():
                shutil.rmtree(dst_dataset)
            shutil.copytree(dataset_dir, dst_dataset)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for run in summary:
        exp_dir = str(run.get("experiment_dir", ""))
        if "FORMAL_FALLBACK_ROOT/" in exp_dir:
            rel = exp_dir.split("FORMAL_FALLBACK_ROOT/", 1)[1]
            run["experiment_dir"] = str(target / Path(rel))
    (target / "summary_formal.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True
