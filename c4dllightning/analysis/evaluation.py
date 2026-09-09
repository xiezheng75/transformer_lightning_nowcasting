import concurrent.futures
import multiprocessing
import os
import numpy as np
import torch
from scipy.integrate import trapezoid
from torch.utils.data import DataLoader
from typing import List, Dict, Union


def _ensure_probabilities(y_pred: np.ndarray) -> np.ndarray:
    """Ensure predictions are in [0, 1]. Apply sigmoid if logits."""
    if y_pred.size == 0:
        return y_pred
    if np.nanmin(y_pred) < 0.0 or np.nanmax(y_pred) > 1.0:
        return 1.0 / (1.0 + np.exp(-y_pred))
    return y_pred


def confusion_matrix(model: torch.nn.Module,
                     batch_gen,
                     dataset: str = 'valid',
                     thresholds: List[float] = [0.5]) -> np.ndarray:
    """计算混淆矩阵 (PyTorch版本)

    Args:
        model: 训练好的PyTorch模型
        batch_gen: 批次数据生成器
        dataset: 数据集类型 ('train'/'valid'/'test')
        thresholds: 分类阈值列表

    Returns:
        np.ndarray: 形状为(2, 2, len(thresholds))的混淆矩阵
    """
    dataset = BatchDataset(batch_gen, dataset=dataset)
    loader = DataLoader(dataset, batch_size=batch_gen.batch_size, shuffle=False)

    num_thresholds = len(thresholds)
    tp = np.zeros(num_thresholds, dtype=np.uint64)
    fp = np.zeros(num_thresholds, dtype=np.uint64)
    fn = np.zeros(num_thresholds, dtype=np.uint64)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            Y_pred = model(X).cpu().numpy()
            Y_pred = _ensure_probabilities(Y_pred)
            Y = Y.cpu().numpy().astype(bool)

            # 多线程计算不同阈值
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i, threshold in enumerate(thresholds):
                    futures.append(executor.submit(
                        _calculate_stats, Y_pred, Y, i, threshold
                    ))
                concurrent.futures.wait(futures)

                for future in futures:
                    i, _tp, _fp, _fn = future.result()
                    tp[i] += _tp
                    fp[i] += _fp
                    fn[i] += _fn

    N = len(dataset) * np.prod(Y_pred.shape[1:])
    tn = N - tp - fp - fn

    return np.array([[[tp, fn], [fp, tn]]]) / N


def _calculate_stats(Y_pred: np.ndarray, Y: np.ndarray,
                     i: int, threshold: float) -> tuple:
    """辅助函数：计算单个阈值的统计量"""
    Y_pred_thresh = (Y_pred >= threshold)
    _tp = np.count_nonzero(Y_pred_thresh & Y)
    _fp = np.count_nonzero(Y_pred_thresh & ~Y)
    _fn = np.count_nonzero(~Y_pred_thresh & Y)
    return (i, _tp, _fp, _fn)


def conf_matrix_models(model: torch.nn.Module,
                       batch_gen,
                       weight_files: List[str],
                       out_dir: str,
                       dataset: str = 'valid') -> None:
    """批量计算多个模型的混淆矩阵"""
    os.makedirs(out_dir, exist_ok=True)
    thresholds = np.arange(0, 1.0001, 0.001)

    for fn in weight_files:
        try:
            model.load_state_dict(torch.load(fn))
            conf_matrix = confusion_matrix(
                model, batch_gen, dataset=dataset, thresholds=thresholds
            )
            fn_root = os.path.splitext(os.path.basename(fn))[0]
            np.save(
                os.path.join(out_dir, f"conf_matrix-{fn_root}.npy"),
                conf_matrix
            )
        except Exception as e:
            print(f"Error processing {fn}: {e}")


def conf_matrix_leadtimes(model: torch.nn.Module,
                          batch_gen,
                          dataset: str = 'valid',
                          thresholds: List[float] = [0.5],
                          num_leadtimes: int = 12) -> np.ndarray:
    """按预测时间步计算混淆矩阵"""
    dataset = BatchDataset(batch_gen, dataset=dataset)
    loader = DataLoader(dataset, batch_size=batch_gen.batch_size, shuffle=False)

    shape = (len(thresholds), num_leadtimes)
    tp = np.zeros(shape, dtype=np.uint64)
    fp = np.zeros(shape, dtype=np.uint64)
    fn = np.zeros(shape, dtype=np.uint64)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            Y_pred = model(X).cpu().numpy()  # (B,T,...)
            Y_pred = _ensure_probabilities(Y_pred)
            Y = Y.cpu().numpy().astype(bool)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for i, threshold in enumerate(thresholds):
                    for t in range(num_leadtimes):
                        futures.append(executor.submit(
                            _calculate_stats,
                            Y_pred[:, t, ...], Y[:, t, ...], i, t, threshold
                        ))
                concurrent.futures.wait(futures)

                for future in futures:
                    i, t, _tp, _fp, _fn = future.result()
                    tp[i, t] += _tp
                    fp[i, t] += _fp
                    fn[i, t] += _fn

    N = len(dataset) * np.prod(Y_pred.shape[2:])
    tn = N - tp - fp - fn

    return np.array([[[tp, fn], [fp, tn]]]) / N


def accuracy(conf_matrix: np.ndarray) -> np.ndarray:
    """计算准确率 (Accuracy)

    Args:
        conf_matrix: 混淆矩阵

    Returns:
        np.ndarray: 准确率
    """
    ((tp, fn), (fp, tn)) = conf_matrix
    return (tp + tn) / (tp + tn + fp + fn)

# 以下指标计算函数与原始版本保持一致 (输入为混淆矩阵)
def precision(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.true_divide(tp, (tp + fp))
        if np.ndim(prec) == 0:
            if not np.isfinite(prec):
                return np.array(1.0)
            return prec
        prec[~np.isfinite(prec)] = 1  # 处理除零情况
    return prec


def recall(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = np.true_divide(tp, (tp + fn))
        rec = np.where(np.isfinite(rec), rec, 0.0)  # Handle division by zero
    return rec


def false_alarm_ratio(conf_matrix: np.ndarray) -> np.ndarray:
    prec = precision(conf_matrix)
    far = 1.0 - prec
    return np.where(np.isfinite(far), far, 0.0)  # Handle edge cases


def intersection_over_union(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.true_divide(tp, (tp + fp + fn))
        iou = np.where(np.isfinite(iou), iou, 0.0)
    return iou


def equitable_threat_score(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        tp_rnd = (tp + fn) * (tp + fp) / (tp + fp + tn + fn)
        ets = np.true_divide(tp - tp_rnd, tp + fp + fn - tp_rnd)
        ets = np.where(np.isfinite(ets), ets, 0.0)
    return ets


def peirce_skill_score(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        pss = np.true_divide(tp * tn - fn * fp, (tp + fn) * (fp + tn))
        pss = np.where(np.isfinite(pss), pss, 0.0)
    return pss


def heidke_skill_score(conf_matrix: np.ndarray) -> np.ndarray:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        hss = np.true_divide(2 * (tp * tn - fn * fp),
                            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        hss = np.where(np.isfinite(hss), hss, 0.0)
    return hss


def f1_score(conf_matrix: np.ndarray) -> np.ndarray:
    """F1 score = Dice coefficient = 2*TP / (2*TP + FP + FN).

    Equivalent to the harmonic mean of precision and recall, and monotonically
    related to CSI by F1 = 2*CSI / (1 + CSI).
    """
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.true_divide(2 * tp, (2 * tp + fp + fn))
        f1 = np.where(np.isfinite(f1), f1, 0.0)
    return f1


def roc_area_under_curve(conf_matrix: np.ndarray) -> float:
    ((tp, fn), (fp, tn)) = conf_matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = np.true_divide(tp, (tp + fn))
        fpr = np.true_divide(fp, (fp + tn))
        tpr = np.where(np.isfinite(tpr), tpr, 0.0)
        fpr = np.where(np.isfinite(fpr), fpr, 0.0)

    # Check if we have valid data for AUC calculation
    if len(np.unique(tpr)) == 1 or len(np.unique(fpr)) == 1:
        return np.nan  # Return NaN when AUC cannot be computed

    return trapezoid(tpr[::-1], x=fpr[::-1])


def pr_area_under_curve(conf_matrix: np.ndarray) -> float:
    prec = precision(conf_matrix)
    rec = recall(conf_matrix)

    # Handle edge cases
    if np.all(rec == 0) or np.all(prec == 1):
        return np.nan

    if (rec[-1] != 0) or (prec[-1] != 1):
        rec = np.hstack((rec, 0.0))
        prec = np.hstack((prec, 1.0))

    return trapezoid(prec[::-1], x=rec[::-1])


# 辅助类
class BatchDataset(torch.utils.data.Dataset):
    """PyTorch Dataset适配器 (与ensemble.py中相同)"""

    def __init__(self, batch_gen, dataset='train'):
        self.batch_gen = batch_gen
        self.dataset = dataset
        self.length = len(batch_gen.time_coords[dataset]) // batch_gen.batch_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pred_batch, target_batch = self.batch_gen.batch(idx, self.dataset)
        X = torch.cat([torch.from_numpy(x) for x in pred_batch], dim=-1)
        Y = torch.cat([torch.from_numpy(y) for y in target_batch], dim=-1)
        return X, Y