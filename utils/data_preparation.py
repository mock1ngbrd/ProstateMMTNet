"""Read case lists and class-balanced training/evaluation splits."""
from pathlib import Path


def _validate_case_ids(case_ids, path):
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {path}.")
    for case_id in case_ids:
        if case_id in {".", ".."} or "/" in case_id or "\\" in case_id or ":" in case_id or case_id.endswith(".npz"):
            raise ValueError(f"Case IDs must be file names without the .npz suffix: {case_id!r}.")


def read_case_ids(path):
    """Read one case ID per nonempty line from a UTF-8 text file."""
    path = Path(path)
    case_ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    _validate_case_ids(case_ids, path)
    return case_ids


def _read_split(path):
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines or lines[0] != "train:" or lines.count("train:") != 1 or lines.count("test:") != 1:
        raise ValueError(f"Expected one train: section followed by one test: section in {path}.")
    boundary = lines.index("test:")
    train_ids, test_ids = lines[1:boundary], lines[boundary + 1:]
    _validate_case_ids(train_ids + test_ids, path)
    return [train_ids, test_ids]


def get_train_test_balance(data_path, fold):
    """Return [train, test] IDs for the positive and negative classes."""
    path = Path(data_path)
    positive = _read_split(path / f"fold_{fold}_positive3.txt")
    negative = _read_split(path / f"fold_{fold}_negative3.txt")
    if set(positive[0] + positive[1]) & set(negative[0] + negative[1]):
        raise ValueError("Positive and negative split files must not share case IDs.")
    return positive, negative
