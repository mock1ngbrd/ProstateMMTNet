"""Checkpoint evaluation and positive-class score export."""
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.ProstateDataset import PD1C_B_E, ToTensor
from networks.model import ProstateMMTNet
from utils.data_preparation import get_train_test_balance
from validation import cal_metrics


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA-enabled PyTorch and an NVIDIA GPU.")
    positive, negative = get_train_test_balance(args.fold_path, args.fold)
    dataset = PD1C_B_E(positive[1] + negative[1], args.data_path, transform=ToTensor())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = ProstateMMTNet(num_channels1=1, num_channels2=args.se_channels, num_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.cuda().eval()
    labels, scores = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation"):
            outputs = model(batch["volume1"].cuda().float(), batch["volume2"].cuda().float())
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            labels.extend(batch["cspca"])
            scores.extend(F.softmax(logits, dim=1)[:, 1].cpu())
    labels, scores = torch.stack(labels), torch.stack(scores)
    metrics = cal_metrics(labels, scores, sensitivity_threshold=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scores.tolist()) + "\n", encoding="utf-8")
    print("AUC={:.4f} F1={:.4f} ACC={:.4f} Sen={:.4f} Spe={:.4f}".format(*metrics[:5]))
    print("Positive ACC={:.4f} Negative ACC={:.4f} Threshold={:.4f}".format(*metrics[5:]))
    print(f"Predictions saved to {args.output}")
