"""Classification metrics and validation checkpoint selection.

Validation sensitivity/specificity use 0.5; accuracy uses the best-F1 threshold.
Testing passes sensitivity_threshold=None to use best-F1 for all these metrics.
"""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_curve, roc_curve
from tqdm import tqdm


def compute_accuracy_by_class(ground_truth, predicted, class_label):
    correct = 0
    total = 0
    for true_label, predicted_label in zip(ground_truth, predicted):
        if true_label == class_label:
            total += 1
            if predicted_label == class_label:
                correct += 1
    if total == 0:
        return 0
    accuracy = correct / total
    return accuracy


def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (sensitivity, specificity)


def cal_metrics(val_y_true, val_y_pred_sm, sensitivity_threshold=0.5):
    fpr, tpr, _ = roc_curve(
        val_y_true.detach().cpu().numpy(),
        val_y_pred_sm.detach().cpu().numpy(),
        pos_label=1,
    )
    precision, recall, thresholds = precision_recall_curve(
        val_y_true.detach().cpu().numpy(),
        val_y_pred_sm.detach().cpu().numpy(),
    )
    precision, recall = (precision[:-1], recall[:-1])
    denominator = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_threshold_index = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    threshold = best_threshold if sensitivity_threshold is None else sensitivity_threshold
    pred_list = [1 if x > threshold else 0 for x in val_y_pred_sm.detach().cpu().numpy()]
    pred_list_bt = [1 if x > best_threshold else 0 for x in val_y_pred_sm.detach().cpu().numpy()]
    acc = accuracy_score(val_y_true.detach().cpu().numpy(), pred_list_bt)
    val_auc = auc(fpr, tpr)
    sensitivity, specificity = calculate_sensitivity_specificity(val_y_true.detach().cpu().numpy(), pred_list)
    pos_acc = compute_accuracy_by_class(val_y_true, pred_list_bt, 1)
    neg_acc = compute_accuracy_by_class(val_y_true, pred_list_bt, 0)
    res = [
        val_auc,
        f1_scores[best_threshold_index],
        acc,
        sensitivity,
        specificity,
        pos_acc,
        neg_acc,
        best_threshold,
    ]
    data = np.array(res)
    data[np.isnan(data)] = 0
    return data.tolist()


def crossval(model, val_dataloader, writer, epoch, exp_save_path, best_auc):
    model.eval()
    val_y_true = []
    val_y_pred = []
    for val_batch in tqdm(val_dataloader, desc='Validation'):
        val_label = val_batch['cspca']
        with torch.no_grad():
            val_bmode = val_batch['volume1'].float().cuda()
            val_se = val_batch['volume2'].float().cuda()
            y = model(val_bmode, val_se)
        if isinstance(y, (list, tuple)):
            out = y[0]
        else:
            out = y
        y_sm = F.softmax(out, dim=1)
        val_y_true.extend(val_label)
        val_y_pred.extend(y_sm[:, 1])
    val_y_true = torch.stack(val_y_true, dim=0)
    val_y_pred = torch.stack(val_y_pred, dim=0)
    metrics3d = cal_metrics(val_y_true, val_y_pred)
    logging.info('val 3d result(%d): \nAUC=%.4f f1_score=%.4f ACC=%.4f Sen=%.4f Spe=%.4f Thr=%.4f' % (
        epoch,
        metrics3d[0],
        metrics3d[1],
        metrics3d[2],
        metrics3d[3],
        metrics3d[4],
        metrics3d[7],
    ))
    logging.info('positive acc: {:.4f} negative acc: {:.4f}'.format(metrics3d[5], metrics3d[6]))
    writer.add_scalar(f'val/auc3d', metrics3d[0], global_step=epoch)
    writer.add_scalar(f'val/acc', metrics3d[2], global_step=epoch)
    val_auc3d = metrics3d[0]
    if val_auc3d >= best_auc[0]:
        best_auc = np.array(metrics3d)
        torch.save(model.module.state_dict(), '{}/ckp_model/best_model.pth'.format(exp_save_path))
    writer.add_scalar(f'val/best_model_result_auc3d', best_auc[0], global_step=epoch)
    return best_auc
