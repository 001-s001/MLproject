import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

from kaggle_kernel_sota.deadline_tools import (
    DEADLINE_RUN_TAG,
    build_deadline_run_plan,
    install_packaged_formal_fallback,
    select_best_runs,
    validate_prediction_file,
)


ROOT = Path(__file__).resolve().parents[1]


class KaggleDeadlineToolsTest(unittest.TestCase):
    def test_deadline_run_plan_contains_only_high_value_kaggle_experiments(self):
        plan = build_deadline_run_plan()

        self.assertEqual(DEADLINE_RUN_TAG, "deadline_sweep")
        self.assertEqual([run["dataset"] for run in plan], ["BCIC2A", "MDD", "SEED", "SLEEP"])
        self.assertTrue(all(run["run_tag"] == DEADLINE_RUN_TAG for run in plan))
        self.assertEqual(plan[0]["epochs"], 200)
        self.assertEqual(plan[0]["patience"], 40)
        self.assertEqual(plan[1]["lr"], 3e-4)
        self.assertEqual(plan[2]["weight_decay"], 1e-3)
        self.assertEqual(plan[2]["label_smoothing"], 0.05)
        self.assertIs(plan[3]["use_class_weight"], True)

    def test_select_best_runs_keeps_formal_fallback_when_deadline_is_worse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_dir = root / "formal_run"
            deadline_dir = root / "deadline_run"
            formal_dir.mkdir()
            deadline_dir.mkdir()
            (formal_dir / "predictions.txt").write_text("0\n1\n", encoding="utf-8")
            (deadline_dir / "predictions.txt").write_text("1\n1\n", encoding="utf-8")

            formal = [{
                "dataset": "MDD",
                "model": "Deep4Net",
                "best_val_acc": 0.8359,
                "run_tag": "formal",
                "experiment_dir": str(formal_dir),
            }]
            deadline = [{
                "dataset": "MDD",
                "model": "Deep4Net",
                "best_val_acc": 0.831,
                "run_tag": DEADLINE_RUN_TAG,
                "experiment_dir": str(deadline_dir),
            }]

            selected = select_best_runs([formal, deadline], min_improvement=0.0001)

        self.assertEqual(selected["MDD"]["run_tag"], "formal")
        self.assertEqual(selected["MDD"]["best_val_acc"], 0.8359)

    def test_select_best_runs_uses_deadline_when_it_beats_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal_dir = root / "formal_run"
            deadline_dir = root / "deadline_run"
            formal_dir.mkdir()
            deadline_dir.mkdir()
            (formal_dir / "predictions.txt").write_text("0\n1\n", encoding="utf-8")
            (deadline_dir / "predictions.txt").write_text("1\n1\n", encoding="utf-8")

            selected = select_best_runs([[
                {
                    "dataset": "SEED",
                    "model": "Deep4Net",
                    "best_val_acc": 0.4244,
                    "run_tag": "formal",
                    "experiment_dir": str(formal_dir),
                },
                {
                    "dataset": "SEED",
                    "model": "Deep4Net",
                    "best_val_acc": 0.431,
                    "run_tag": DEADLINE_RUN_TAG,
                    "experiment_dir": str(deadline_dir),
                },
            ]])

        self.assertEqual(selected["SEED"]["run_tag"], DEADLINE_RUN_TAG)
        self.assertEqual(selected["SEED"]["experiment_dir"], str(deadline_dir))

    def test_select_best_runs_ignores_runs_without_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_dir = root / "valid"
            missing_dir = root / "missing"
            valid_dir.mkdir()
            missing_dir.mkdir()
            (valid_dir / "predictions.txt").write_text("0\n", encoding="utf-8")

            selected = select_best_runs([[
                {
                    "dataset": "BCIC2A",
                    "model": "ShallowFBCSPNet",
                    "best_val_acc": 0.49,
                    "run_tag": "formal",
                    "experiment_dir": str(valid_dir),
                },
                {
                    "dataset": "BCIC2A",
                    "model": "ShallowFBCSPNet",
                    "best_val_acc": 0.99,
                    "run_tag": DEADLINE_RUN_TAG,
                    "experiment_dir": str(missing_dir),
                },
            ]])

        self.assertEqual(selected["BCIC2A"]["experiment_dir"], str(valid_dir))

    def test_validate_prediction_file_checks_count_and_integer_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred = Path(tmp) / "predictions.txt"
            pred.write_text("0\n1\n2\n", encoding="utf-8")

            self.assertEqual(validate_prediction_file(pred, expected_count=3, num_classes=3), [])

            pred.write_text("0\n3\nbad\n", encoding="utf-8")
            errors = validate_prediction_file(pred, expected_count=3, num_classes=3)

        self.assertIn("line 2 label 3 outside [0, 2]", errors)
        self.assertIn("line 3 is not an integer: bad", errors)

    def test_deadline_tools_imports_from_kaggle_kernel_directory(self):
        command = (
            "import sys; "
            "from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd() / 'src')); "
            "import deadline_tools; "
            "assert deadline_tools.DEADLINE_RUN_TAG == 'deadline_sweep'"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=ROOT / "kaggle_kernel_sota",
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_install_packaged_formal_fallback_copies_summary_and_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "formal_fallback"
            fallback_run = package / "BCIC2A" / "ShallowFBCSPNet" / "formal_bcic2a"
            fallback_run.mkdir(parents=True)
            (fallback_run / "predictions.txt").write_text("0\n1\n", encoding="utf-8")
            (fallback_run / "config.json").write_text("{}", encoding="utf-8")
            (package / "summary_formal.json").write_text(json.dumps([{
                "dataset": "BCIC2A",
                "model": "ShallowFBCSPNet",
                "best_val_acc": 0.4944,
                "run_tag": "formal",
                "experiment_dir": "FORMAL_FALLBACK_ROOT/BCIC2A/ShallowFBCSPNet/formal_bcic2a",
            }]), encoding="utf-8")

            results_dir = root / "_results_braindecode"
            installed = install_packaged_formal_fallback(package, results_dir)

            copied_pred = results_dir / "BCIC2A" / "ShallowFBCSPNet" / "formal_bcic2a" / "predictions.txt"
            summary = json.loads((results_dir / "summary_formal.json").read_text(encoding="utf-8"))

            self.assertTrue(installed)
            self.assertTrue(copied_pred.exists())
            self.assertEqual(summary[0]["experiment_dir"], str(copied_pred.parent))


if __name__ == "__main__":
    unittest.main()
