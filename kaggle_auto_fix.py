from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT / "kaggle_kernel_sota"
NOTEBOOK = KERNEL_DIR / "train_braindecode_sota_kaggle.ipynb"
KERNEL_REF = "llllyu8568/ml-project-sota-braindecode"
LOG_DIR = ROOT / "kaggle_debug_logs"
OUTPUT_DIR = ROOT / "kaggle_outputs"


class StopAutomation(RuntimeError):
    pass


def kaggle_exe() -> str:
    candidate = Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "Scripts" / "kaggle.exe"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("kaggle")
    if found:
        return found
    raise StopAutomation("Kaggle CLI not found. Install it with: python -m pip install --user kaggle")


def run_kaggle(args: list[str], timeout: int = 120, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    cmd = [kaggle_exe(), *args]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise StopAutomation(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}")
    return proc


def load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def save_notebook(nb: dict) -> None:
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def set_preflight_mode(enabled: bool) -> bool:
    nb = load_notebook()
    changed = False
    desired = f"PREFLIGHT_ONLY = {str(enabled)}"
    other = f"PREFLIGHT_ONLY = {str(not enabled)}"
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "PREFLIGHT_ONLY =" in src:
            new_src = src.replace(other, desired)
            if new_src != src:
                cell["source"] = new_src.splitlines(keepends=True)
                changed = True
    if changed:
        save_notebook(nb)
    return changed


def ensure_notebook_guards() -> bool:
    """Apply idempotent static fixes needed before any Kaggle run."""
    nb = load_notebook()
    changed = False

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "# Formal experiment configuration" in src:
            if "PREFLIGHT_ONLY =" not in src:
                src = src.replace(
                    "SYNC_TO_DRIVE = False\n",
                    "SYNC_TO_DRIVE = False\n\n"
                    "# Kaggle quota guard: preflight first, and never train on CPU by accident.\n"
                    "PREFLIGHT_ONLY = True\n"
                    "REQUIRE_GPU = True\n",
                )
                changed = True
            if "REQUIRE_GPU =" not in src:
                src = src.replace("PREFLIGHT_ONLY = True\n", "PREFLIGHT_ONLY = True\nREQUIRE_GPU = True\n")
                changed = True
            cell["source"] = src.splitlines(keepends=True)

        if "def load_dataset(data_root, dataset_name):" in src and "using inline fallback" in src:
            if '"n_train": len(X_train)' not in src:
                src = src.replace(
                    '"sfreq": sfreq, "window_sec": n_time / sfreq,\n        }',
                    '"sfreq": sfreq, "window_sec": n_time / sfreq,\n'
                    '            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),\n'
                    "        }",
                )
                changed = True
            if "return self.model(x.unsqueeze(1))" in src:
                src = src.replace("return self.model(x.unsqueeze(1))", "return self.model(x)")
                changed = True
            cell["source"] = src.splitlines(keepends=True)
        if "def train_one_experiment(" in src and "exp_dir.mkdir(parents=True, exist_ok=True)" not in src:
            src = src.replace(
                '    num_classes = metadata["num_classes"]\n',
                '    exp_dir = Path(exp_dir)\n'
                '    exp_dir.mkdir(parents=True, exist_ok=True)\n'
                '    num_classes = metadata["num_classes"]\n',
                1,
            )
            cell["source"] = src.splitlines(keepends=True)
            changed = True

    if "# Kaggle preflight:" not in json.dumps(nb, ensure_ascii=False):
        raise StopAutomation("Notebook is missing the Kaggle preflight cell. Re-run the notebook patch step before automation.")

    if changed:
        save_notebook(nb)
    return changed


def push_kernel() -> str:
    proc = run_kaggle(["kernels", "push", "-p", str(KERNEL_DIR), "--accelerator", "gpu"], timeout=600)
    return proc.stdout


def kernel_status() -> str:
    proc = run_kaggle(["kernels", "status", KERNEL_REF], timeout=60)
    match = re.search(r'status "([^"]+)"', proc.stdout)
    return match.group(1) if match else proc.stdout.strip()


def kernel_logs() -> str:
    proc = run_kaggle(["kernels", "logs", KERNEL_REF], timeout=120, check=False)
    return proc.stdout


def save_logs(logs: str, label: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{stamp}_{label}.log"
    path.write_text(logs, encoding="utf-8")
    return path


def classify_logs(logs: str) -> str:
    if "GPU is required for this Kaggle run" in logs or "CUDA available: False" in logs:
        return "missing_gpu"
    if "requires sm_70 or newer" in logs or "not compatible with the current PyTorch installation" in logs:
        return "unsupported_gpu"
    if "KeyError: 'n_train'" in logs or "KeyError: 'n_val'" in logs or "KeyError: 'n_test'" in logs:
        return "missing_metadata_counts"
    if "EinopsError: Shape mismatch" in logs and "800 != 1" in logs:
        return "wrapper_shape"
    if "Parent directory" in logs and "does not exist" in logs and "_results_braindecode/ablation" in logs:
        return "missing_ablation_dir"
    if "DATA_ROOT_OVERRIDE set but invalid" in logs or "Cannot locate dataset root" in logs:
        return "data_root"
    if "No module named 'braindecode'" in logs or "Could not find a version that satisfies the requirement braindecode" in logs:
        return "braindecode_dependency"
    if "PREFLIGHT_ONLY=True; downstream training cells will be skipped" in logs and "Traceback" not in logs:
        return "preflight_complete"
    if "All Braindecode formal experiments finished" in logs:
        return "formal_complete"
    return "unknown"


def apply_known_fix(kind: str) -> bool:
    if kind in {"missing_metadata_counts", "wrapper_shape", "missing_ablation_dir"}:
        return ensure_notebook_guards()
    if kind in {"data_root", "braindecode_dependency"}:
        return ensure_notebook_guards()
    return False


def wait_for_terminal_status(poll_seconds: int, max_wait_minutes: int) -> tuple[str, str, Path]:
    deadline = time.time() + max_wait_minutes * 60
    last_status = ""
    while time.time() < deadline:
        status = kernel_status()
        if status != last_status:
            print(f"[status] {status}")
            last_status = status
        if status.endswith(".ERROR") or status.endswith(".COMPLETE") or status in {"ERROR", "COMPLETE"}:
            logs = kernel_logs()
            path = save_logs(logs, status.replace(".", "_"))
            return status, logs, path
        time.sleep(poll_seconds)
    logs = kernel_logs()
    path = save_logs(logs, "timeout")
    raise StopAutomation(f"Timed out waiting for Kaggle kernel after {max_wait_minutes} minutes. Logs: {path}")


def download_outputs() -> list[Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    run_kaggle(["kernels", "output", KERNEL_REF, "-p", str(OUTPUT_DIR), "-o"], timeout=600, check=False)
    return [p for p in OUTPUT_DIR.rglob("*") if p.is_file()]


def output_success(files: list[Path]) -> bool:
    names = [str(p.relative_to(OUTPUT_DIR)).replace("\\", "/") for p in files]
    has_summary = any(name.endswith("_results_braindecode/summary_formal.json") or name.endswith("summary_formal.json") for name in names)
    has_csv = any(name.endswith("_results_braindecode/experiments_summary_formal.csv") or name.endswith("experiments_summary_formal.csv") for name in names)
    has_predictions = any("/predictions/" in name and name.endswith(".txt") for name in names)
    return has_summary and has_csv and has_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Low-quota Kaggle auto-fix runner for the SOTA Braindecode notebook.")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-minutes", type=int, default=240)
    parser.add_argument("--start-formal", action="store_true", help="Skip preflight mode and start a formal training push.")
    args = parser.parse_args()

    ensure_notebook_guards()
    set_preflight_mode(not args.start_formal)

    for round_no in range(1, args.max_rounds + 1):
        print(f"\n=== Kaggle auto-fix round {round_no}/{args.max_rounds} ===")
        print(push_kernel())
        status, logs, log_path = wait_for_terminal_status(args.poll_seconds, args.max_wait_minutes)
        print(f"[logs] {log_path}")
        kind = classify_logs(logs)
        print(f"[classification] {kind}")

        if kind == "missing_gpu":
            raise StopAutomation(
                "Kaggle did not allocate a GPU. Open the notebook Settings, set Accelerator=GPU, "
                "then rerun this script. Formal training was not started."
            )
        if kind == "unsupported_gpu":
            raise StopAutomation(
                "Kaggle allocated a P100/old GPU that is incompatible with the current PyTorch build. "
                "Select T4 instead of P100, then rerun this script. Formal training was not started."
            )
        if kind == "preflight_complete":
            print("[preflight] Passed. Switching to formal training mode.")
            set_preflight_mode(False)
            continue
        if kind == "formal_complete" or status.endswith(".COMPLETE") or status == "COMPLETE":
            files = download_outputs()
            if output_success(files):
                print(f"[success] Downloaded {len(files)} output files to {OUTPUT_DIR}")
                return 0
            raise StopAutomation(f"Kernel completed, but required prediction/summary files were not found in {OUTPUT_DIR}.")

        fixed = apply_known_fix(kind)
        if fixed:
            print(f"[fix] Applied known fix for {kind}; pushing another version.")
            continue
        raise StopAutomation(f"Unrecognized or unfixable Kaggle failure ({kind}). Latest logs: {log_path}")

    raise StopAutomation(f"Reached max rounds ({args.max_rounds}) without success.")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StopAutomation as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        raise SystemExit(2)
