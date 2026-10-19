"""Check CLI options and split handling without machine-learning dependencies."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from utils.cli import build_test_parser, build_train_parser, validate_inputs
from utils.data_preparation import get_train_test_balance, read_case_ids


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prostate_mmt_checks_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.splits, self.volumes, self.masks = [self.root / n for n in ("splits", "volumes", "masks")]
        for path in (self.splits, self.volumes, self.masks):
            path.mkdir()
        self.positive = self.splits / "fold_sample_positive3.txt"
        self.negative = self.splits / "fold_sample_negative3.txt"
        self.positive.write_text("train:\np_annotated\np_unannotated\np_excluded\ntest:\np_test\n", encoding="utf-8")
        self.negative.write_text("train:\nn_train\ntest:\nn_test\n", encoding="utf-8")
        self.annotations = self.splits / "annotated.txt"
        self.annotations.write_text("p_annotated\n", encoding="utf-8")
        for case in ("p_annotated", "p_unannotated", "p_excluded", "p_test", "n_train", "n_test"):
            (self.volumes / f"{case}.npz").touch()
        (self.masks / "p_annotated.npz").touch()
        self.checkpoint = self.root / "model.pth"
        self.checkpoint.touch()
        self.common = ["--data-path", str(self.volumes), "--fold-path", str(self.splits), "--fold", "sample"]

    def training_args(self, *extra):
        return build_train_parser().parse_args(self.common + [
            "--mask-path", str(self.masks), "--annotated-cases", str(self.annotations),
            "--output-dir", str(self.root / "runs"), *extra,
        ])

    def test_valid_training_does_not_create_outputs(self):
        args = self.training_args()
        validate_inputs(args, training=True)
        self.assertFalse(args.output_dir.exists())

    def test_evaluation_channels(self):
        args = build_test_parser().parse_args(self.common + ["--checkpoint", str(self.checkpoint), "--se-channels", "1"])
        validate_inputs(args)
        self.assertEqual(args.se_channels, 1)

    def test_case_order(self):
        positive, negative = get_train_test_balance(self.splits, "sample")
        self.assertEqual(positive, [["p_annotated", "p_unannotated", "p_excluded"], ["p_test"]])
        self.assertEqual(negative, [["n_train"], ["n_test"]])

    def test_split_leakage(self):
        self.positive.write_text("train:\np_test\ntest:\np_test\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            get_train_test_balance(self.splits, "sample")

    def test_class_overlap(self):
        self.negative.write_text("train:\np_annotated\ntest:\nn_test\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "must not share"):
            get_train_test_balance(self.splits, "sample")

    def test_annotation_membership(self):
        self.annotations.write_text("p_test\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "positive training split"):
            validate_inputs(self.training_args(), training=True)

    def test_existing_experiment_is_preserved(self):
        args = self.training_args()
        output = args.output_dir / args.exp_name
        output.mkdir(parents=True)
        sentinel = output / "keep.txt"
        sentinel.write_text("preserve", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            validate_inputs(args, training=True)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")

    def test_required_input_files(self):
        (self.masks / "p_annotated.npz").unlink()
        with self.assertRaisesRegex(ValueError, "Missing mask"):
            validate_inputs(self.training_args(), training=True)
        (self.masks / "p_annotated.npz").touch()
        (self.volumes / "p_test.npz").unlink()
        with self.assertRaisesRegex(ValueError, "Missing volume"):
            validate_inputs(self.training_args(), training=True)

    def test_exclusion_list(self):
        excluded = self.splits / "excluded.txt"
        excluded.write_text("p_excluded\n", encoding="utf-8")
        validate_inputs(self.training_args("--exclude-cases", str(excluded)), training=True)

    def test_empty_training_stream(self):
        self.annotations.write_text("p_annotated\np_unannotated\np_excluded\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unannotated positive"):
            validate_inputs(self.training_args(), training=True)

    def test_invalid_options(self):
        for flag, value in [("--epochs", "0"), ("--learning-rate", "nan"), ("--workers", "-1"), ("--exp-name", "../outside")]:
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.training_args(flag, value)

    def test_case_list_encoding_and_paths(self):
        self.annotations.write_text("p_annotated\n", encoding="utf-8-sig")
        self.assertEqual(read_case_ids(self.annotations), ["p_annotated"])
        self.annotations.write_text("../outside\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file names"):
            read_case_ids(self.annotations)

    def test_entry_point_help(self):
        project = Path(__file__).resolve().parents[1]
        for script in ("train.py", "test.py"):
            result = subprocess.run([sys.executable, "-B", str(project / script), "--help"], cwd=project, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--data-path", result.stdout)


if __name__ == "__main__":
    unittest.main()
