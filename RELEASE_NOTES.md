# Release preparation

ProstateMMTNet is the descriptive implementation name for the multi-modality,
multi-task framework in "Computer-Aided Diagnosis of Clinically Significant
Prostate Cancer in Transrectal Ultrasound".

## Source cleanup

- Use the public entry points `train.py` and `test.py`, with implementations in
  `training.py` and `evaluation.py`.
- Keep only ProstateMMTNet and its model blocks. Removed unused architecture
  variants, preprocessing routines, datasets, augmentations, validation branches,
  debugging examples, commented experiments, and unused imports.
- Replace wildcard imports with explicit imports and fix testing to import the
  model from `networks.model`.
- Remove private filesystem defaults, study case IDs, and automatic experiment
  deletion. Data, mask, annotation, checkpoint, and output paths have generic
  relative defaults and can be overridden through command-line options.
- Use grouped, documented CLI options with positive-number checks. Help and
  input-configuration validation run without importing PyTorch.
- Validate split membership and required files. Annotation IDs must belong to
  the positive training partition; evaluation cases cannot enter this list.
- Apply `--workers` to all training loaders and use a picklable worker initializer.
- Remove dependencies used only by discarded helpers. Remaining packages are
  PyTorch, torchvision, TensorBoard, NumPy, SciPy, scikit-learn, and tqdm.
- Share classification metrics between validation and testing while retaining
  their original threshold rules. Exclude the precision-recall endpoint without
  a matching threshold and handle zero denominators locally.
- Retain two prototype feature scales as confirmed by the author. Guard missing
  prototype updates and consistency-loss logging; keep the model's functional
  import at module scope.

The retained model classes are structurally unchanged. Registered parameter
names are preserved. The original research workspace outside this folder was
not edited. `source_manifest.json` records source origins where applicable and
current release hashes; generated support files have no original source entry.

## Remaining release information

- Confirm the tested Python, PyTorch, torchvision, and CUDA versions.
- Confirm the original preprocessing and normalization protocol.
- Provide approved data access instructions and split files if appropriate.
- Review the existing checkpoint-selection and threshold-selection protocol
  documented in the README before reporting independent test performance.
- Supply the final paper citation, license, and GitHub destination.
- Run GPU training and inference with representative data and checkpoints.

The shipped split files contain fictional IDs. No study data, original patient
split files, spreadsheets, experiment outputs, checkpoints, or paper PDF are
included.
