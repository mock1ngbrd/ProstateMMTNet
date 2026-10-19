"""Command-line options and input checks without machine-learning dependencies."""
import argparse
from pathlib import Path

from utils.data_preparation import get_train_test_balance, read_case_ids


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def positive_float(value):
    value = float(value)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def experiment_name(value):
    if value in {".", ".."} or not value or any(c in value for c in '/\\:'):
        raise argparse.ArgumentTypeError("must be a single folder name")
    return value


def _common_options(parser):
    data = parser.add_argument_group("data")
    data.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/volumes"),
        help="Folder containing case NPZ files.",
    )
    data.add_argument(
        "--fold-path",
        type=Path,
        default=Path("splits"),
        help="Folder containing class split files.",
    )
    data.add_argument(
        "--fold",
        default="example",
        type=experiment_name,
        help="Name in fold_<name>_{positive,negative}3.txt.",
    )
    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--gpu", default="0", help="CUDA device IDs, for example 0 or 0,1.")
    runtime.add_argument(
        "--se-channels",
        type=positive_int,
        default=3,
        help="Number of strain-elastography input channels.",
    )
    return data, runtime


def build_train_parser():
    parser = argparse.ArgumentParser(
        description="Train ProstateMMTNet for binary csPCa classification and segmentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data, runtime = _common_options(parser)
    data.add_argument(
        "--mask-path",
        type=Path,
        default=Path("data/masks"),
        help="Folder containing annotated-case mask NPZ files.",
    )
    data.add_argument(
        "--annotated-cases",
        type=Path,
        default=Path("splits/annotated_example.txt"),
        help="Text file listing annotated positive training case IDs.",
    )
    data.add_argument(
        "--exclude-cases",
        type=Path,
        help="Optional text file listing positive training IDs to exclude.",
    )
    runtime.add_argument(
        "--workers",
        type=nonnegative_int,
        default=2,
        help="Data-loader worker processes; use 0 to load in the main process.",
    )
    output = parser.add_argument_group("outputs")
    output.add_argument(
        "--exp-name",
        type=experiment_name,
        default="prostate_mmt_run",
        help="New experiment folder name; existing folders are never overwritten.",
    )
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Parent folder for experiments.",
    )
    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=positive_int, default=200, help="Training epochs.")
    training.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1e-4,
        help="Initial AdamW learning rate.",
    )
    training.add_argument(
        "--positive-batch-size",
        type=positive_int,
        default=1,
        help="Unannotated positive cases per batch.",
    )
    training.add_argument(
        "--negative-batch-size",
        type=positive_int,
        default=1,
        help="Negative cases per batch.",
    )
    training.add_argument(
        "--annotated-batch-size",
        type=positive_int,
        default=2,
        help="Annotated positive cases per batch.",
    )
    training.add_argument(
        "--val-batch-size",
        type=positive_int,
        default=8,
        help="Validation cases per batch.",
    )
    training.add_argument(
        "--max-steps-per-epoch",
        type=positive_int,
        default=100,
        help="Upper limit on optimizer steps in an epoch.",
    )
    training.add_argument("--seed", type=nonnegative_int, default=2000, help="Random seed.")
    training.add_argument(
        "--log-every",
        type=positive_int,
        default=2,
        help="Training steps between log messages.",
    )
    training.add_argument(
        "--validate-every",
        type=positive_int,
        default=1,
        help="Epochs between validation passes.",
    )
    training.add_argument(
        "--save-every",
        type=positive_int,
        default=100,
        help="Epochs between periodic checkpoints.",
    )
    return parser


def build_test_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a ProstateMMTNet checkpoint on the test: split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common_options(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/prostate_mmt_run/ckp_model/best_model.pth"),
        help="Path to a model state-dictionary checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=8,
        help="Evaluation cases per batch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/predictions.json"),
        help="JSON file for positive-class scores.",
    )
    return parser


def _check_volumes(data_path, case_ids):
    if not data_path.is_dir():
        raise ValueError(f"Data folder does not exist: {data_path}")
    for case_id in case_ids:
        path = data_path / f"{case_id}.npz"
        if not path.is_file():
            raise ValueError(f"Missing volume file: {path}")


def validate_inputs(args, training=False):
    """Check split membership and required files before creating outputs."""
    positive, negative = get_train_test_balance(args.fold_path, args.fold)
    if not positive[1] or not negative[1]:
        raise ValueError("Evaluation requires at least one positive and one negative test case.")
    required_ids = positive[1] + negative[1]
    if training:
        annotated = read_case_ids(args.annotated_cases)
        excluded = read_case_ids(args.exclude_cases) if args.exclude_cases else []
        if not annotated:
            raise ValueError("The annotated case list must not be empty.")
        if not set(annotated + excluded).issubset(positive[0]):
            raise ValueError("Annotated and excluded cases must belong to the positive training split.")
        if set(annotated) & set(excluded):
            raise ValueError("Annotated and excluded case lists must be disjoint.")
        unannotated = set(positive[0]) - set(annotated) - set(excluded)
        if not unannotated or not negative[0]:
            raise ValueError("Training requires unannotated positive cases and negative cases.")
        required_ids += sorted(unannotated) + annotated + negative[0]
        if not args.mask_path.is_dir():
            raise ValueError(f"Mask folder does not exist: {args.mask_path}")
        for case_id in annotated:
            if not (args.mask_path / f"{case_id}.npz").is_file():
                raise ValueError(f"Missing mask for annotated case: {case_id}")
        if (args.output_dir / args.exp_name).exists():
            raise ValueError("Experiment folder already exists; choose another --exp-name.")
    elif not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    _check_volumes(args.data_path, required_ids)
