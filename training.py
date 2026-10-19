"""Training loop and multi-scale prototype losses for ProstateMMTNet."""
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader.ProstateDataset import PD1C_B_E, PD1C_B_E_mask, ToTensor
from dataloader.data_aug import MirrorTransform, RandomRotateTransform
from networks.model import ProstateMMTNet
from utils.data_preparation import get_train_test_balance, read_case_ids
from validation import crossval


def prepare_experiment(path, args):
    """Create a new output folder without deleting previous experiments."""
    path.mkdir(parents=True, exist_ok=False)
    (path / "ckp_model").mkdir()
    for name in ("train.py", "training.py"):
        shutil.copy(Path(__file__).with_name(name), path / name)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        handlers=[logging.FileHandler(path / "log.txt"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    logging.info("Arguments: %s", vars(args))


def seed_worker(worker_id, seed):
    random.seed(seed + worker_id)


def create_model(args):
    model = ProstateMMTNet(num_channels1=1, num_channels2=args.se_channels, num_classes=2)
    model = nn.DataParallel(model)
    return model.cuda()


def reproduce(seed1):
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)


def train(args):
    warmup_epochs = 30
    warmup_epochs2 = 40
    ema_alpha = 0.9
    lambda_align = 5
    batch_size_labeled = args.annotated_batch_size
    lambda_seg = 0.5
    lambda_seg_neg = 0.1
    lambda_contrast = 10
    scale = 2
    reproduce(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA-enabled PyTorch and an NVIDIA GPU.')
    exp_save_path = Path(args.output_dir) / args.exp_name
    prepare_experiment(exp_save_path, args)
    writer = SummaryWriter(log_dir=str(exp_save_path))
    pos_list, neg_list = get_train_test_balance(args.fold_path, args.fold)
    annotated_ids = read_case_ids(args.annotated_cases)
    excluded_ids = read_case_ids(args.exclude_cases) if args.exclude_cases else []
    train_pos_list = [case for case in pos_list[0] if case not in annotated_ids and case not in excluded_ids]
    train_dataset_positive_nobox = PD1C_B_E(
        train_pos_list,
        args.data_path,
        transform=transforms.Compose([
            RandomRotateTransform(mask_prefix=False, angle_range=(-10, 10), p_per_sample=0.2),
            MirrorTransform(mask_prefix=False, axes=(-3, -2, -1)),
            ToTensor(mask_prefix=False),
        ]),
    )
    train_dataset_positive_box = PD1C_B_E_mask(
        annotated_ids,
        args.data_path,
        args.mask_path,
        transform=transforms.Compose([
            RandomRotateTransform(mask_prefix=True, angle_range=(-10, 10), p_per_sample=0.2),
            MirrorTransform(mask_prefix=True, axes=(-3, -2, -1)),
            ToTensor(mask_prefix=True),
        ]),
    )
    train_dataset_negative = PD1C_B_E(
        neg_list[0],
        args.data_path,
        transform=transforms.Compose([
            RandomRotateTransform(mask_prefix=False, angle_range=(-10, 10), p_per_sample=0.2),
            MirrorTransform(mask_prefix=False, axes=(-3, -2, -1)),
            ToTensor(mask_prefix=False),
        ]),
    )
    val_datasets = PD1C_B_E(
        pos_list[1] + neg_list[1],
        args.data_path,
        transform=transforms.Compose([ToTensor(mask_prefix=False)]),
    )
    worker_init_fn = partial(seed_worker, seed=args.seed)
    train_dataloader_positive_nobox = DataLoader(
        train_dataset_positive_nobox,
        batch_size=args.positive_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    train_dataloader_positive_box = DataLoader(
        train_dataset_positive_box,
        batch_size=batch_size_labeled,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    train_dataloader_negative = DataLoader(
        train_dataset_negative,
        batch_size=args.negative_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(val_datasets, batch_size=args.val_batch_size, shuffle=True)
    model = create_model(args)
    model_py_path = model.module.get_model_py_path()
    shutil.copy(model_py_path, os.path.join(exp_save_path, os.path.basename(model_py_path)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    celoss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent
    n_total_iter = 0
    best_metrics = np.zeros(8)
    memory_bank = {'foreground': [None] * scale, 'background': [None] * scale}
    for epoch in range(1, args.epochs + 1):
        lr = poly_lr(epoch, args.epochs, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        i_batch = 0
        loss_epoch = ce_loss_epoch = 0
        train_prefetcher_pos_nobox = iter(train_dataloader_positive_nobox)
        train_prefetcher_pos_box = iter(train_dataloader_positive_box)
        train_prefetcher_neg = iter(train_dataloader_negative)
        pos_batch_nobox = next(train_prefetcher_pos_nobox, None)
        pos_batch_box = next(train_prefetcher_pos_box, None)
        neg_batch = next(train_prefetcher_neg, None)
        y_true = []
        y_pred_sm = []
        model.train()
        while neg_batch is not None and pos_batch_nobox is not None and (i_batch < args.max_steps_per_epoch):
            optimizer.zero_grad()
            if pos_batch_box is not None:
                data, has_mask = (pos_batch_box, True)
            else:
                data, has_mask = (pos_batch_nobox, False)
            pos_bmode, pos_se, pos_label = (data['volume1'], data['volume2'], data['cspca'])
            neg_bmode, neg_se, neg_label = (neg_batch['volume1'], neg_batch['volume2'], neg_batch['cspca'])
            Bp = pos_bmode.size(0)
            Bn = neg_bmode.size(0)
            train_bmodes = torch.cat([pos_bmode, neg_bmode], dim=0).cuda().float()
            train_se = torch.cat([pos_se, neg_se], dim=0).cuda().float()
            train_labels = torch.cat([pos_label, neg_label], dim=0).cuda()
            out_cls, out_seg, feats_list = model(train_bmodes, train_se, return_feats=True)
            loss = 0
            cls_loss = celoss(out_cls, train_labels)
            loss += cls_loss
            if epoch > warmup_epochs:
                mask_neg = torch.zeros(Bn, *out_seg.shape[2:]).cuda().long()
                neg_seg_loss = celoss(out_seg[Bp:], mask_neg)
                loss += lambda_seg * lambda_seg_neg * neg_seg_loss
                if has_mask:
                    mask_pos = data['mask'].cuda()
                    mask_pos = F.interpolate(mask_pos.float(), out_seg.shape[2:]).squeeze(1).long()
                    pos_seg_loss = celoss(out_seg[:Bp], mask_pos)
                    loss += lambda_seg * pos_seg_loss
            with torch.no_grad():
                probs = F.softmax(out_cls, dim=1)
                conf, pr = probs.max(1)
                reliable = (pr == train_labels) & (conf > 0.8)
            if epoch > warmup_epochs2 and reliable.any():
                with torch.no_grad():
                    feats_pos = [f[:Bp] for f in feats_list]
                    feats_neg = [f[Bp:] for f in feats_list]
                    if has_mask:
                        mask_pos = data['mask'].cuda().long().squeeze(1)
                        protos = compute_multiscale_prototypes(
                            feats_pos, mask_pos, correct_mask=reliable[:Bp], num_classes=2,
                        )
                        update_memory_bank(memory_bank, protos, 'foreground', reliable[:Bp], ema_alpha)
                        update_memory_bank(memory_bank, protos, 'background', reliable[:Bp], ema_alpha)
                    mask_bg = torch.zeros(Bn, *feats_neg[0].shape[2:], device=feats_neg[0].device).long()
                    protos_bg = compute_multiscale_prototypes(feats_neg, mask_bg, reliable[Bp:], num_classes=2)
                    update_memory_bank(memory_bank, protos_bg, 'background', reliable[Bp:], ema_alpha)
            has_none = any((p is None for values in memory_bank.values() for p in values))
            cons_loss = None
            if epoch > warmup_epochs2 and (not has_mask) and (not has_none):
                with torch.no_grad():
                    feats_pos = [f[:Bp] for f in feats_list]
                proto_logits = prototype_segmentation_multiscale(feats_pos, memory_bank)
                seg_soft = F.softmax(out_seg[:Bp], dim=1)
                ups = []
                for pl in proto_logits:
                    if pl is None:
                        continue
                    ups.append(F.interpolate(
                        pl,
                        size=seg_soft.shape[2:],
                        mode='trilinear',
                        align_corners=False,
                    ))
                if ups:
                    proto_f = torch.stack(ups).mean(0)
                    cons_loss = mse_loss(F.softmax(proto_f.detach(), 1), seg_soft)
                    loss += lambda_align * cons_loss
                pseudo_correct = torch.ones((Bp,), dtype=torch.bool).cuda()
                pseudo_protos = compute_multiscale_prototypes(
                    feats_pos,
                    seg_soft.argmax(1),
                    correct_mask=pseudo_correct,
                    num_classes=2,
                )
                cont_loss = contrastive_loss(pseudo_protos, memory_bank, margin=0.8)
                loss += lambda_contrast * cont_loss
            y_true.extend(train_labels.cpu().detach())
            y_pred_sm.extend(probs[:, 1].cpu().detach())
            loss_epoch += loss.item()
            ce_loss_epoch += cls_loss.item()
            loss.backward()
            optimizer.step()
            pos_batch_nobox = next(train_prefetcher_pos_nobox, None)
            if has_mask:
                pos_batch_box = next(train_prefetcher_pos_box, None)
            neg_batch = next(train_prefetcher_neg, None)
            n_total_iter += 1
            i_batch += 1
            if n_total_iter % args.log_every == 0:
                log_msg = '[Epoch: %4d/%d] [Train index: %2d/%d] [loss: %.4f] [cls_loss: %.4f]' % (
                    epoch,
                    args.epochs,
                    i_batch,
                    len(train_dataloader_negative),
                    loss.item(),
                    cls_loss.item(),
                )
                if epoch > warmup_epochs:
                    log_msg += ' [neg_seg_loss: %.4f]' % neg_seg_loss.item()
                    if has_mask:
                        log_msg += ' [pos_seg_loss: %.4f]' % pos_seg_loss.item()
                if epoch > warmup_epochs2 and (not has_mask) and (not has_none):
                    log_msg += ' [cont_loss: %.4f]' % cont_loss.item()
                    if cons_loss is not None:
                        log_msg += ' [cons_loss: %.4f]' % cons_loss.item()
                logging.info(log_msg)
        y_true = torch.stack(y_true, dim=0)
        y_pred_sm = torch.stack(y_pred_sm, dim=0)
        fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), y_pred_sm.cpu().numpy(), pos_label=1)
        train_auc = auc(fpr, tpr)
        writer.add_scalar(
            'Loss/loss',
            loss_epoch / len(train_dataloader_negative),
            global_step=epoch,
        )
        writer.add_scalar(
            'Loss/cls_loss',
            ce_loss_epoch / len(train_dataloader_negative),
            global_step=epoch,
        )
        writer.add_scalar('Loss/lr', lr, global_step=epoch)
        writer.add_scalar('train/auc', train_auc, epoch)
        logging.info('epoch: {}  train auc: {:.4f}'.format(epoch, train_auc))
        if epoch % args.validate_every == 0:
            best_metrics = crossval(
                model,
                val_dataloader,
                writer,
                epoch,
                exp_save_path,
                best_metrics,
            )
        if epoch % args.save_every == 0:
            torch.save(
                model.module.state_dict(),
                '{}/ckp_model/model_{}.pth'.format(exp_save_path, epoch),
            )
    logging.info('Best result/fold')
    logging.info('AUC:{:.4f} F1:{:.4f} ACC:{:.4f} Sen:{:.4f} Spe:{:.4f} PosACC:{:.4f} NegACC:{:.4f}'.format(
        best_metrics[0],
        best_metrics[1],
        best_metrics[2],
        best_metrics[3],
        best_metrics[4],
        best_metrics[5],
        best_metrics[6],
    ))
    torch.save(model.module.state_dict(), '{}/ckp_model/last_model.pth'.format(exp_save_path))
    writer.close()


def compute_multiscale_prototypes(feats_list, masks, correct_mask, num_classes=2, eps=1e-06):
    """Compute class prototypes from reliable samples at each decoder scale."""
    idx = correct_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    masks_f = masks[idx]
    prototypes_per_scale = []
    for feats in feats_list:
        feats_f = feats[idx]
        _, _, D, H, W = feats_f.shape
        m_resized = F.interpolate(masks_f.unsqueeze(1).float(), size=(D, H, W), mode='nearest').squeeze(1)
        protos = []
        for cls in range(num_classes):
            mask_cls = (m_resized == cls).float()
            cnt = mask_cls.sum()
            if cnt < 1e-06:
                protos.append(None)
                continue
            masked_feats = feats_f * mask_cls.unsqueeze(1)
            sum_feat = masked_feats.sum(dim=[0, 2, 3, 4])
            proto = sum_feat / (cnt + eps)
            protos.append(proto)
        prototypes_per_scale.append(protos)
    return prototypes_per_scale


def update_memory_bank(memory_bank, new_protos, key, correct, alpha=0.9):
    """Update class prototypes with an EMA when reliable samples are available."""
    if new_protos is None or not correct.any():
        return
    cls_idx = 1 if key == 'foreground' else 0
    for t in range(len(new_protos)):
        p_new = new_protos[t][cls_idx]
        if p_new is None:
            continue
        p_old = memory_bank[key][t]
        if p_old is None:
            memory_bank[key][t] = p_new.clone()
        else:
            memory_bank[key][t] = alpha * p_old + (1 - alpha) * p_new


def prototype_segmentation_multiscale(feats_list, memory_bank):
    """Return per-scale cosine-similarity logits for background and foreground."""
    protos_bg = memory_bank['background']
    protos_fg = memory_bank['foreground']
    proto_logits = []
    for t, feats in enumerate(feats_list):
        B, C, D, H, W = feats.shape
        p_bg = protos_bg[t]
        p_fg = protos_fg[t]
        if p_bg is None or p_fg is None:
            proto_logits.append(None)
            continue
        protos = torch.stack([p_bg, p_fg], dim=0)
        f_n = F.normalize(feats, p=2, dim=1)
        p_n = F.normalize(protos, p=2, dim=1)
        flat = f_n.permute(0, 2, 3, 4, 1).reshape(-1, C)
        sims = flat @ p_n.t()
        logit = sims.view(B, D, H, W, 2).permute(0, 4, 1, 2, 3)
        proto_logits.append(logit)
    return proto_logits


def contrastive_loss(pseudo_protos, memory_bank, margin=0.2):
    """Penalize prototype similarities below the requested cosine margin."""
    losses = []
    for t, (p_bg, p_fg) in enumerate(pseudo_protos):
        mem_bg = memory_bank['background'][t]
        mem_fg = memory_bank['foreground'][t]
        for p, m in [(p_bg, mem_bg), (p_fg, mem_fg)]:
            if p is None or m is None:
                continue
            cos = F.cosine_similarity(p.unsqueeze(0), m.unsqueeze(0), dim=1).clamp(-1, 1)
            losses.append(F.relu(margin - cos))
    if not losses:
        return memory_bank['foreground'][0].new_zeros(())
    return torch.stack(losses).mean()
