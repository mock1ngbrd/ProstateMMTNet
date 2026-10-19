# ProstateMMTNet

Code for **Computer-Aided Diagnosis of Clinically Significant Prostate Cancer in
Transrectal Ultrasound**.

**ProstateMMTNet** means **Prostate Multi-Modality Multi-Task Network**. This
implementation combines B-mode and strain elastography (SE) inputs for joint
csPCa classification and segmentation. It uses dual 3D ResNet-34 streams,
cross-modal attention gates, and a context-aware multi-scale classifier.
Multi-scale prototype-guided learning is implemented in the training loop.

## Files

```text
train.py                 Training command-line entry point
training.py              Training loop and prototype learning
test.py                 Evaluation command-line entry point
evaluation.py            Checkpoint evaluation and score export
validation.py            Classification metrics and checkpoint selection
networks/model.py        ProstateMMTNet and the blocks it uses
dataloader/              Paired NPZ loading and spatial augmentation
utils/cli.py             Command-line options and input checks
utils/data_preparation.py Case-list and split readers
splits/                  Fictional example split and annotation lists
tests/                   Command-line and data-configuration checks
```

No study data, original case lists, trained checkpoints, or patient spreadsheets
are included. Replace the fictional example IDs with IDs for your own data.

## Environment

Training and evaluation require an NVIDIA GPU and CUDA-enabled PyTorch with a
compatible torchvision installation. Install the dependencies in your Python
environment:

```bash
python -m pip install -r requirements.txt
```

The paper's exact Python, PyTorch, torchvision, and CUDA versions still need to
be confirmed. Requirements are therefore unpinned. Full training and inference
have not been validated in this workspace, which lacks PyTorch and study data.

Both help commands work before installing the machine-learning dependencies:

```bash
python train.py --help
python test.py --help
```

## Input data

Store one `<case_id>.npz` file per case:

| Key | Stored shape | Meaning |
| --- | --- | --- |
| `volume1` | `(Z, Y, X)` | B-mode volume |
| `volume2` | `(Z, Y, X, C)` | Aligned SE volume; default `C=3` |
| `label` | Scalar | Benign/malignant indicator; `0` forces the csPCa target to `0` |
| `CsPCa` | Scalar | Binary classification target for cases with nonzero `label` |

The loader transposes these into `(1, X, Y, Z)` and `(C, X, Y, Z)`. The model's
attention layers expect `(X, Y, Z) = (144, 144, 200)`. Arrays must already be
aligned and resampled; the loader does not resize or normalize them. The original
preprocessing protocol still needs author confirmation.

For annotated positive training cases, provide matching mask NPZ files with a
binary `mask` array of stored shape `(Z, Y, X)`:

```text
data/
    volumes/
        <case_id>.npz
    masks/
        <annotated_positive_case_id>.npz
```

## Split and annotation lists

Create two UTF-8 files named `fold_<name>_positive3.txt` and
`fold_<name>_negative3.txt`. The `3` suffix is retained for compatibility with the
original split format. Each file contains:

```text
train:
training_case_001
training_case_002
test:
evaluation_case_001
```

IDs must match NPZ filenames without the `.npz` suffix. Positive cases have csPCa
target `1`; negative cases have target `0`. Training/evaluation cases and the two
classes must be disjoint. Both classes must be present in the evaluation split.

The default annotation list is `splits/annotated_example.txt`. Replace its
fictional IDs with your annotated positive training IDs, one per line, or pass
a different file with `--annotated-cases`. Each listed case must have a corresponding mask file.
See `splits/annotated_example.txt` for the format. To exclude additional positive
training cases, optionally pass `--exclude-cases` with another case-list file.
No source editing is needed. Training requires at least one annotated positive,
one unannotated positive, and one negative training case.

The commands check case membership, duplicate IDs, and required input files
before creating an experiment directory.

## Training

Run commands from the project folder. All options have defaults, defined in
`utils/cli.py`. `train.py` parses those defaults and any command-line overrides,
then passes the resulting `args` object to `training.train(args)` in
`training.py`. An option omitted from a command still receives its default value;
you do not need to list every training setting.

### Run with defaults

After preparing real volumes and masks, replacing the example split IDs, and
updating the annotation list, use:

```bash
python train.py
```

The default input layout is:

```text
data/volumes/                         Case NPZ files
data/masks/                           Annotated positive mask NPZ files
splits/fold_example_positive3.txt      Positive training/evaluation IDs
splits/fold_example_negative3.txt      Negative training/evaluation IDs
splits/annotated_example.txt           Annotated positive training IDs
```

The supplied split and annotation files contain fictional IDs. The command
requires your actual NPZ files and matching case lists. Relative paths are
resolved from the directory where you run the command.

### Override selected defaults

For example, change the epoch count, learning rate, GPU, and experiment name:

```bash
python train.py --epochs 100 --learning-rate 0.00005 --gpu 0 --exp-name experiment_02
```

For data stored elsewhere, override only the relevant input locations:

```bash
python train.py --data-path ./my_data/volumes --mask-path ./my_data/masks --fold-path ./my_splits --fold custom --annotated-cases ./my_splits/annotated.txt
```

All other settings keep their defaults. `--fold custom` selects
`fold_custom_positive3.txt` and `fold_custom_negative3.txt` inside `--fold-path`.
If you change the split folder or fold name, also provide the annotation list
for that training split; its default path remains `splits/annotated_example.txt`.

### Training options

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-path` | `data/volumes` | Folder containing case NPZ files |
| `--fold-path` | `splits` | Folder containing class split files |
| `--fold` | `example` | Name in fold_<name>_{positive,negative}3.txt |
| `--gpu` | `0` | Visible CUDA device IDs; for example `0` or `0,1` |
| `--se-channels` | `3` | Number of strain-elastography input channels |
| `--mask-path` | `data/masks` | Folder containing annotated-case mask NPZ files |
| `--annotated-cases` | `splits/annotated_example.txt` | Text file listing annotated positive training case IDs |
| `--exclude-cases` | None (disabled) | Optional text file listing positive training IDs to exclude |
| `--workers` | `2` | Data-loader worker processes; use 0 to load in the main process |
| `--exp-name` | `prostate_mmt_run` | New experiment folder name; existing folders are never overwritten |
| `--output-dir` | `runs` | Parent folder for experiments |
| `--epochs` | `200` | Training epochs |
| `--learning-rate` | `0.0001` | Initial AdamW learning rate |
| `--positive-batch-size` | `1` | Unannotated positive cases per batch |
| `--negative-batch-size` | `1` | Negative cases per batch |
| `--annotated-batch-size` | `2` | Annotated positive cases per batch |
| `--val-batch-size` | `8` | Validation cases per batch |
| `--max-steps-per-epoch` | `100` | Upper limit on optimizer steps in an epoch |
| `--seed` | `2000` | Random seed |
| `--log-every` | `2` | Training steps between log messages |
| `--validate-every` | `1` | Epochs between validation passes |
| `--save-every` | `100` | Epochs between periodic checkpoints |

Use `python train.py --help` to see the same descriptions and defaults in the
terminal. Both entry points use `--se-channels` for the SE input channel count.

The original two-scale prototype learning, loss weights, and warm-up schedule
are retained: segmentation supervision starts after epoch 30, and prototype
updates/consistency losses start after epoch 40. An epoch stops when a training
stream ends or the configured step limit is reached.

Outputs are written to `runs/prostate_mmt_run/`:

- `log.txt` and TensorBoard event files;
- `ckp_model/best_model.pth`, selected by validation-partition AUC;
- periodic checkpoints and `ckp_model/last_model.pth`;
- copies of the training entry point, training implementation, and model source.

Choose a new experiment name for each run. Existing experiment folders are
never deleted or overwritten by the training command.

```bash
tensorboard --logdir ./runs
```

## Checkpoint evaluation

After training the default experiment, evaluate its best checkpoint with:

```bash
python test.py
```

To evaluate a different run or change the output file:

```bash
python test.py --checkpoint ./runs/experiment_02/ckp_model/best_model.pth --output ./results/experiment_02.json
```

The data, split, GPU, and SE-channel options have the same defaults as training.
For a separate evaluation split, override `--fold-path` and `--fold`.
The default checkpoint path stays tied to `prostate_mmt_run`; specify
`--checkpoint` whenever you use a different experiment name or output directory.

### Evaluation options

| Option | Default | Purpose |
| --- | --- | --- |
| `--data-path` | `data/volumes` | Folder containing case NPZ files |
| `--fold-path` | `splits` | Folder containing class split files |
| `--fold` | `example` | Name in fold_<name>_{positive,negative}3.txt |
| `--gpu` | `0` | Visible CUDA device IDs; for example `0` or `0,1` |
| `--se-channels` | `3` | Number of strain-elastography input channels |
| `--checkpoint` | `runs/prostate_mmt_run/ckp_model/best_model.pth` | Path to a model state-dictionary checkpoint |
| `--batch-size` | `8` | Evaluation cases per batch |
| `--output` | `results/predictions.json` | JSON file for positive-class scores |

Testing prints AUC, F1, accuracy, sensitivity, specificity, per-class accuracy,
and the selected threshold. The JSON file contains positive-class scores in the
order of positive `test:` IDs followed by negative `test:` IDs. IDs and labels
are not included in that JSON. `--checkpoint` accepts a model state dictionary
saved by this training implementation. Use the same `--se-channels` value as
training.

## Evaluation protocol

The original training code uses the split files' `test:` partition for per-epoch
validation and checkpoint selection. Evaluating that same partition includes
the effects of model selection. For independent final evaluation, supply a
separate held-out split through `--fold-path` and `--fold`.

Testing selects a threshold by maximizing F1 on the evaluation labels.
Threshold-dependent metrics therefore reflect threshold selection on that
partition. Validation sensitivity/specificity use threshold 0.5, while accuracy
uses the best-F1 threshold. The original strict `score > threshold` classification
rule is retained. These evaluation rules have not been redesigned during cleanup.

## Checks and release status

Run the dependency-free configuration checks with:

```bash
python -m unittest discover -s tests -v
```

Model class definitions were compared structurally during cleanup to ensure
that removing unused variants did not change the architecture or parameter
names. GPU training, tensor-level model execution, and real checkpoint inference
remain unverified. See `RELEASE_NOTES.md` for changes and remaining release
information. The complete paper citation and license are pending confirmation.
