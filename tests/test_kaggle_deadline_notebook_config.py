import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "kaggle_kernel_sota" / "train_braindecode_sota_kaggle.ipynb"


def notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


class KaggleDeadlineNotebookConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = notebook_source()

    def test_kaggle_notebook_uses_deadline_sweep_plan(self):
        self.assertIn("from deadline_tools import", self.source)
        self.assertIn("build_deadline_run_plan", self.source)
        self.assertIn("install_packaged_formal_fallback", self.source)
        self.assertIn("def _define_inline_deadline_tools", self.source)
        self.assertIn("FORMAL_FALLBACK_PAYLOAD", self.source)
        self.assertIn('RUN_TAG = "deadline_sweep"', self.source)
        self.assertIn("DEADLINE_RUN_PLAN", self.source)

    def test_deadline_plan_keeps_formal_fallback_names_available(self):
        self.assertIn('FALLBACK_RUN_TAGS = ["formal", "deadline_sweep"]', self.source)
        self.assertIn("FORMAL_FALLBACK_DIR", self.source)
        self.assertIn("install_packaged_formal_fallback(FORMAL_FALLBACK_DIR, RESULTS_DIR_BASE)", self.source)
        self.assertIn("summary_formal.json", self.source)
        self.assertIn("summary_deadline_sweep.json", self.source)

    def test_training_loop_supports_class_weight_for_sleep(self):
        self.assertIn("use_class_weight=False", self.source)
        self.assertIn("compute_class_weight", self.source)
        self.assertIn("nn.CrossEntropyLoss(weight=class_weight_tensor", self.source)
        self.assertIn('"use_class_weight": run_cfg["use_class_weight"]', self.source)

    def test_prediction_export_selects_best_valid_run_not_latest_run(self):
        self.assertIn("select_best_runs", self.source)
        self.assertIn("validate_prediction_file", self.source)
        self.assertIn("selected_runs = select_best_runs", self.source)
        self.assertNotIn("latest = runs[0]", self.source)

    def test_kaggle_deadline_run_skips_slow_ablation_and_debug_import(self):
        self.assertIn("RUN_ABLATION = False", self.source)
        self.assertIn("if PREFLIGHT_ONLY or not RUN_ABLATION:", self.source)
        self.assertNotIn("import data_adapter\n", self.source)


if __name__ == "__main__":
    unittest.main()
