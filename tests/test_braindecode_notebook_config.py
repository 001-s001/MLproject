import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "train_braindecode.ipynb"


def notebook_source() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n\n".join("".join(cell.get("source", [])) for cell in nb["cells"])


class BraindecodeNotebookConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = notebook_source()

    def test_formal_training_excludes_chinese(self):
        match = re.search(r"DATASETS\s*=\s*\[(.*?)\]", self.source, re.S)
        self.assertIsNotNone(match, "DATASETS must be defined in the notebook")
        datasets_literal = match.group(1)
        self.assertIn('"BCIC2A"', datasets_literal)
        self.assertIn('"MDD"', datasets_literal)
        self.assertIn('"SEED"', datasets_literal)
        self.assertIn('"SLEEP"', datasets_literal)
        self.assertNotIn('"CHINESE"', datasets_literal)

    def test_formal_outputs_are_separate_from_smoke_outputs(self):
        self.assertIn('RUN_TAG = "formal"', self.source)
        self.assertIn('experiments_summary_formal.csv', self.source)
        self.assertIn('summary_formal.json', self.source)
        self.assertRegex(self.source, r"exp_name\s*=.*RUN_TAG")

    def test_existing_smoke_runs_do_not_block_formal_training(self):
        self.assertIn("SKIP_EXISTING = False", self.source)
        self.assertIn("RUN_TAG in d.name", self.source)

    def test_training_loop_uses_scheduler_and_regularization_controls(self):
        self.assertIn("torch.optim.lr_scheduler.CosineAnnealingLR", self.source)
        self.assertIn("label_smoothing=label_smoothing", self.source)
        self.assertIn("torch.nn.utils.clip_grad_norm_", self.source)

    def test_comparison_ignores_excluded_datasets(self):
        self.assertIn("EXCLUDED_DATASETS", self.source)
        self.assertIn("if ds in EXCLUDED_DATASETS", self.source)

    def test_reports_are_generated_from_best_checkpoint_predictions(self):
        load_idx = self.source.index("model.load_state_dict(checkpoint['model_state_dict'])")
        report_idx = self.source.index("classification_report(")
        best_eval_idx = self.source.index("best_val_preds")
        self.assertGreater(best_eval_idx, load_idx)
        self.assertLess(best_eval_idx, report_idx)
        self.assertIn("all_val_preds = best_val_preds", self.source)
        self.assertIn("all_val_labels = best_val_labels", self.source)

    def test_next_round_lightweight_hyperparameter_overrides_are_configured(self):
        self.assertIn('"BCIC2A": {"lr": 3e-4, "epochs": 120, "patience": 25}', self.source)
        self.assertIn('"SEED":   {"lr": 5e-4, "epochs": 100, "patience": 20', self.source)
        self.assertIn('"SLEEP":  {"lr": 5e-4, "epochs": 100, "patience": 20', self.source)
        self.assertIn('"label_smoothing": 0.02', self.source)
        self.assertIn('"weight_decay": 5e-4', self.source)


if __name__ == "__main__":
    unittest.main()
