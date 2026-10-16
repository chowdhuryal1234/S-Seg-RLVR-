"""Prevent generated reports and notebook outputs from returning to Git."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.check_repository_contents import check_index, content_errors


def notebook(**updates):
    cell = {"cell_type": "code", "source": ["print('hello')\n"], "metadata": {}, "outputs": [], "execution_count": None}
    cell.update(updates)
    return json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}).encode()


class RepositoryContentTests(unittest.TestCase):
    def test_source_config_readmes_and_workflow_are_allowed(self):
        for name in ("nucleus_rl/train.py", "scripts/render_report.py", "tests/test_train.py", "protocols/cohort.json", "baselines/iris/README.md", "README.md", "LICENSE", "requirements-gpu-lock.txt", ".github/workflows/check.yml"):
            with self.subTest(name=name):
                self.assertEqual(content_errors(name, b"source"), [])

    def test_generated_paths_documents_and_binary_outputs_are_rejected(self):
        for name in ("reports/README.md", "runs/a/result.json", "data/cohort.json", "checkpoints/model.pt", "scripts/results/scores.json", "MEETING_BRIEF.md", "protocols/experiment_report.md", "baselines/iris/AUDIT.md", "figures/chart.png", "scripts/slide.pdf", "scripts/result.html", "protocols/monuseg_official30_grpo_v1.json"):
            with self.subTest(name=name):
                self.assertTrue(content_errors(name, b"content"))
        self.assertTrue(content_errors("scripts/hidden.py", b"a\0b"))
        self.assertTrue(content_errors("tests/hidden.json", b"\xff\xfe"))

    def test_notebook_source_and_explanations_allowed_outputs_rejected(self):
        self.assertEqual(content_errors("baselines/iris/b.ipynb", notebook()), [])
        for changes in ({"outputs": [{"output_type": "stream", "text": "result"}]}, {"execution_count": 1}, {"metadata": {"colab": {"outputId": "abc"}}}, {"attachments": {"plot.png": {"image/png": "x"}}}):
            with self.subTest(changes=changes):
                self.assertTrue(content_errors("baselines/iris/b.ipynb", notebook(**changes)))
        self.assertEqual(content_errors("baselines/iris/b.ipynb", notebook(cell_type="markdown", source=["## How to evaluate\n"])), [])
        self.assertTrue(content_errors("baselines/iris/b.ipynb", notebook(cell_type="markdown", source=["# Baseline Results\nPQ = 0.9"])))
        for raw in (b"not json", b"[]", b"null", b'{"cells": {}}'):
            self.assertTrue(content_errors("baselines/iris/b.ipynb", raw))

    def test_index_is_checked_instead_of_working_tree_or_untracked_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "baselines").mkdir()
            path = root / "baselines/example.ipynb"
            path.write_bytes(notebook(outputs=[{"output_type": "stream", "text": "secret result"}]))
            subprocess.run(["git", "add", "baselines/example.ipynb"], cwd=root, check=True)
            path.write_bytes(notebook())
            self.assertTrue(check_index(root), "Unstaged cleanup must not hide staged outputs")
            subprocess.run(["git", "add", "baselines/example.ipynb"], cwd=root, check=True)
            (root / "reports").mkdir()
            (root / "reports/local.json").write_text("{}")
            self.assertEqual(check_index(root), [], "Untracked local evidence must be ignored")
            subprocess.run(["git", "add", "reports/local.json"], cwd=root, check=True)
            self.assertTrue(check_index(root))


if __name__ == "__main__":
    unittest.main()
